from engram import EngramConfig
import engram
from engram_qwen import EngramQwenForCausalLM, set_skip_engram
from dataclasses import dataclass, field
import logging
import os
import pathlib
import shutil
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

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    engram_warmup_steps: int = field(default=0, metadata={"help": "Steps to keep engram gates at 1."})
    engram_soft_constraint_steps: int = field(default=0, metadata={"help": "Steps to keep engram gates at min 0.1 after warmup."})
    resume_path: Optional[str] = field(default=None, metadata={"help": "Path to checkpoint to resume from. If None, starts a new run."})
    engram_vocab_size: List[int] = field(default_factory=lambda: [2048, 256], metadata={"help": "List of vocab sizes for engram layers"})
    engram_layer_ids: List[int] = field(default_factory=lambda: [13, 17], metadata={"help": "List of layer indices to apply engram"})


@dataclass
class DataArguments:
    data_path: str = field(
        default="/data3/biomed/*.parquet", metadata={"help": "Path to the training data or dataset name."}
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


class SavePolicyCallback(TrainerCallback):
    """
    Implements a custom saving policy:
    1. Trainer handles the "latest" checkpoints (rolling 2 checkpoints every 1000 steps).
    2. This callback creates a PERMANENT backup every 5000 steps that Trainer won't delete.
    """
    def on_save(self, args, state, control, **kwargs):
        # Trigger every 5000 steps (excluding step 0 if it saves there)
        if state.global_step > 0 and state.global_step % 5000 == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            backup_dir = os.path.join(args.output_dir, f"permanent-checkpoint-{state.global_step}")
            
            if os.path.exists(checkpoint_dir) and not os.path.exists(backup_dir):
                rank0_print(f"Dataset protection: Creating permanent backup for step {state.global_step}...")
                try:
                    # Use hardlinks to save space and time (Linux/Unix only)
                    shutil.copytree(checkpoint_dir, backup_dir, copy_function=os.link)
                    rank0_print(f"  Backup created at: {backup_dir}")
                except Exception as e:
                    rank0_print(f"  Warning: Failed to create hardlink backup: {e}")
                    # Fallback to normal copy (slower)
                    try:
                        shutil.copytree(checkpoint_dir, backup_dir)
                        rank0_print(f"  Standard copy created instead.")
                    except Exception as e2:
                         rank0_print(f"  Error: Failed standard copy too: {e2}")


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
        training_args.output_dir = os.path.dirname(model_args.resume_path)
    else:
        import datetime
        run_name = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        training_args.output_dir = os.path.join(base_output_dir, run_name)
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

    # Load dataset
    rank0_print(f"Loading files from {data_args.data_path}")
    ds = load_dataset(
        "parquet",
        data_files=data_args.data_path, # "/data3/biomed/*.parquet"
        split="train" 
    )

    # Filter base
    rank0_print("Filtering dataset for language='en' and domain='biomedical'...")
    ds = ds.filter(
        lambda x: x.get("language") == "en" and x.get("domain") == "biomedical"
    )

    # Split train/eval by score
    rank0_print("Splitting by educational_score: Train (>4.0), Eval (<4.0)...")
    
    # Train: > 4.0
    train_dataset = ds.filter(lambda x: x.get("educational_score", 0) > 4.0)

    # Eval: < 4.0 (sample 200)
    eval_candidates = ds.filter(lambda x: x.get("educational_score", 0) < 4.0)
    
    # Validation info
    rank0_print(f"Total High Quality (Train) samples: {len(train_dataset)}")
    rank0_print(f"Total Low Quality (Eval Pool) samples: {len(eval_candidates)}")
    
    if len(train_dataset) > 0:
        rank0_print("First train example keys:", train_dataset[0].keys())
        if "text" in train_dataset[0]:
            rank0_print("First train example text Preview:", train_dataset[0]["text"][:500])

    # Sample Eval
    if len(eval_candidates) > 200:
        eval_dataset = eval_candidates.shuffle(seed=42).select(range(200))
    else:
        eval_dataset = eval_candidates
        rank0_print(f"Warning: Only {len(eval_dataset)} samples found for evaluation (< 4.0). Using all.")
    
    rank0_print(f"Final Train size: {len(train_dataset)}, Final Eval size: {len(eval_dataset)}")

    def tokenize_function(examples):
        texts = [t + tokenizer.eos_token for t in examples["text"]]
        return tokenizer(
            texts, truncation=True, max_length=training_args.model_max_length
        )

    with training_args.main_process_first(desc="dataset map tokenization"):
        train_dataset = train_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=train_dataset.column_names,
            num_proc=32  # Speed up with multiprocessing
        )
        eval_dataset = eval_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=eval_dataset.column_names,
            num_proc=32  # Speed up with multiprocessing
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- 4. Start Trainer ---
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
