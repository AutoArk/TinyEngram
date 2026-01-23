from engram import EngramConfig
import engram
from engram_qwen import EngramQwenForCausalLM, set_skip_engram
from dataclasses import dataclass, field
import logging
import os
import pathlib
from typing import Dict, Optional, List
import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import AutoTokenizer
from transformers import Trainer, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling

from transformers.integrations import deepspeed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from datasets import load_dataset, Dataset

from transformers import TrainerCallback

local_rank = None

def smart_load_dataset(path, name=None, split=None, **kwargs):
    prioritize_local = os.environ.get("PRIORITY_LOCAL", "False").lower() in ("true", "1", "yes")

    if prioritize_local:
        try:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            ds = load_dataset(path, name, split=split, **kwargs)
            del os.environ["HF_DATASETS_OFFLINE"]
            return ds
        except Exception:
             if "HF_DATASETS_OFFLINE" in os.environ:
                del os.environ["HF_DATASETS_OFFLINE"]
    
    return load_dataset(path, name, split=split, **kwargs)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    # This should now point to the base multimodal model, e.g., /data/model/ark_audio_tts
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    engram_warmup_steps: int = field(default=0, metadata={"help": "Steps to keep engram gates at 1."})
    engram_soft_constraint_steps: int = field(default=0, metadata={"help": "Steps to keep engram gates at min 0.1 after warmup."})
    resume_path: Optional[str] = field(default=None, metadata={"help": "Path to checkpoint to resume from. If None, starts a new run."})
    engram_vocab_size: List[int] = field(default_factory=lambda: [2048, 256], metadata={"help": "List of vocab sizes for engram layers"})
    engram_layer_ids: List[int] = field(default_factory=lambda: [13, 17], metadata={"help": "List of layer indices to apply engram"})


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data or dataset name."}
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "Path to the evaluation data or dataset name."},
    )
    data_config: str = field(
        default=None,
        metadata={"help": "Dataset configuration name (e.g. 'main' for gsm8k)."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


class EngramStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        engram.set_global_step(state.global_step)


def train():
    global local_rank
    set_skip_engram(False)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if (
        getattr(training_args, "deepspeed", None)
        and int(os.environ.get("WORLD_SIZE", 1)) == 1
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    # --- Custom Output Directory & Resume Logic ---
    base_output_dir = training_args.output_dir
    
    if model_args.resume_path and os.path.exists(model_args.resume_path):
        rank0_print(f"Resuming training from checkpoint: {model_args.resume_path}")
        training_args.resume_from_checkpoint = model_args.resume_path
        # When resuming, use the parent directory of the checkpoint as the output_dir
        # Structure: /path/to/base/run_name/checkpoint-XXX
        training_args.output_dir = os.path.dirname(model_args.resume_path)
    else:
        # New Run: Create a subdirectory with timestamp
        import datetime
        run_name = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        # training_args.output_dir (base) + / + run_name
        training_args.output_dir = os.path.join(base_output_dir, run_name)
        
        # Align tensorboard logs to be under base_output_dir/runs/run_name
        # Note: Trainer appends 'runs/...' relative to logging_dir if not specified, 
        # or logging_dir itself. 
        # To get structure: base/runs/run_name/events..., we set logging_dir to that path.
        training_args.logging_dir = os.path.join(base_output_dir, "runs", run_name)
        
        rank0_print(f"Starting new training run.")
        rank0_print(f"  Output Dir : {training_args.output_dir}")
        rank0_print(f"  Logging Dir: {training_args.logging_dir}")

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 is incompatible with QLoRA.")

    model_load_kwargs = {
        "low_cpu_mem_usage": not deepspeed.is_deepspeed_zero3_enabled(),
    }

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # --- 1. Load Tokenizers ---
    rank0_print("Loading tokenizers...")

    # This is the main tokenizer for text, used by the Trainer and the model.
    rank0_print(f"model_args.model_name_or_path is {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        rank0_print("Warning: pad_token_id is not set. Setting it to eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- 2. Load Model ---
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    model = EngramQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            if training_args.use_lora and lora_args.q_lora
            else None
        ),
        attn_implementation="eager",
        engram_cfg=EngramConfig(
            tokenizer_name_or_path=model_args.model_name_or_path,
            engram_vocab_size=model_args.engram_vocab_size,
            layer_ids=model_args.engram_layer_ids,
            warmup_steps=model_args.engram_warmup_steps,
            soft_constraint_steps=model_args.engram_soft_constraint_steps,
        ),
        # attn_implementation="flash_attention_2",
        **model_load_kwargs,
    )

    for name, param in model.named_parameters():
        if "engram" not in name:
            param.requires_grad = False
            print(f"Freezing {name}")
        else:
            param.requires_grad = True
            print(f"Unfreezing {name}")
            if "value_proj" in name:
                torch.nn.init.zeros_(param)

    model.enable_input_require_grads()

    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # --- 3. Setup Data Pipeline ---
    rank0_print("Setting up data pipeline...")

    def format_text(example):
        # GSM8K style
        if "question" in example and "answer" in example and "options" not in example:
            return f"Question: {example['question']}\nAnswer: {example['answer']}"
        
        # MedQA style (Question + Options + Answer Key)
        if "question" in example and "options" in example and ("answer" in example or "answer_idx" in example):
            question = example["question"]
            options = example["options"]
            answer = example.get("answer", example.get("answer_idx"))
            
            # Format options
            if isinstance(options, dict):
                option_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
            elif isinstance(options, list): # Sometimes options are list
                # Assume list like ["Option A", "Option B"...] or custom logic
                # For simplified logic, try to match A, B, C, D if len <= 26
                option_str = "\n".join([f"{chr(65+i)}. {v}" for i, v in enumerate(options)])
            else:
                option_str = str(options)
                
            return f"Question: {question}\nOptions:\n{option_str}\nAnswer: {answer}"

        # Generic Instruction/Input/Output
        if "instruction" in example and "output" in example:
            prompt = example["instruction"]
            if example.get("input", ""):
                prompt += f"\nInput: {example['input']}"
            return f"Question: {prompt}\nAnswer: {example['output']}"
            
        # Fallback
        return str(example)

    def tokenize_function(examples):
        # Detect keys to determine dataset type for the whole batch
        keys = examples.keys()
        
        texts = []
        # Since examples is a dict of lists, verify the length of one field
        # to know batch size
        num_examples = len(next(iter(examples.values())))
        
        for i in range(num_examples):
            # Reconstruct single item
            ex = {k: examples[k][i] for k in keys}
            text = format_text(ex) + tokenizer.eos_token
            texts.append(text)

        return tokenizer(
            texts, truncation=True, max_length=training_args.model_max_length
        )

    # Load the dataset
    if not data_args.data_path:
        raise ValueError("data_path must be specified.")

    rank0_print(f"Loading dataset from: {data_args.data_path} with config {data_args.data_config}")
    if data_args.data_path.endswith(".json") or data_args.data_path.endswith(".jsonl"):
        train_dataset = smart_load_dataset(
            "json", data_files=data_args.data_path, split="train"
        )
    else:
        # Load from hub or local path using config
        train_dataset = smart_load_dataset(
            data_args.data_path, data_args.data_config, split="train"
        )

    train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=train_dataset.column_names
    )

    eval_dataset = None
    if data_args.eval_data_path:
        rank0_print(f"Loading eval dataset from: {data_args.eval_data_path} with config {data_args.data_config}")
        if data_args.eval_data_path.endswith(
            ".json"
        ) or data_args.eval_data_path.endswith(".jsonl"):
            eval_dataset = smart_load_dataset(
                "json", data_files=data_args.eval_data_path, split="train"
            )
        else:
            # Generic load with config
            eval_dataset = smart_load_dataset(
                data_args.eval_data_path, data_args.data_config, split="test"
            )

        if eval_dataset:
            eval_dataset = eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
            )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- 4. Start Trainer ---
    # The Trainer will automatically handle passing the correct inputs to the model.
    # Columns in the dataset not in the model's forward signature (like 'speaker_embs')
    # will be automatically ignored during the forward pass.
    # To USE speaker_embs, you must modify the model's forward() method to accept it.
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EngramStepCallback],
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        # LoRA with DeepSpeed checkpointing has known issues, so we start fresh.
        # The original check for non-LoRA resume is kept.
        if (
            list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
            and not training_args.use_lora
        ):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()

        safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir,
            bias=lora_args.lora_bias,
        )


if __name__ == "__main__":
    train()
