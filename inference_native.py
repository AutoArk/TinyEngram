"""Optional frozen-checkpoint CPU inference through the meft-memory backend.

No default training/evaluation dependency changes. Fixed-length greedy decode,
unpadded CPU inputs only; no CUDA offload, sampling, beams or EOS stopping.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="existing native bundle, or a fresh output path with --checkpoint")
    parser.add_argument("--checkpoint", type=Path,
                        help="full local trained safetensors checkpoint to convert")
    parser.add_argument("--engram-config", type=Path,
                        help="original EngramConfig JSON; otherwise checkpoint/engram_config.json")
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-bytes", type=int, default=16 * 1024**2,
                        help="decoded native cache budget per Engram layer; not total model RAM")
    parser.add_argument("--query-chunk-rows", type=int, default=65536)
    args = parser.parse_args()
    if args.max_new_tokens < 0 or min(args.threads, args.query_chunk_rows) < 1 or args.cache_bytes < 0:
        parser.error("invalid decode/thread/cache/query limits")
    if args.engram_config and not args.checkpoint:
        parser.error("--engram-config requires --checkpoint")
    if args.checkpoint and args.bundle.exists():
        parser.error("conversion requires a fresh --bundle directory")
    if not args.checkpoint and not args.bundle.is_dir():
        parser.error("provide an existing --bundle or use --checkpoint to create one")
    try:
        import torch
        from transformers import AutoTokenizer
        from meft_memory import export_tinyengram_checkpoint, load_tinyengram_checkpoint
    except ImportError:
        parser.error("install the optional meft-memory[tinyengram] backend; see README")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    upstream = Path(__file__).resolve().parent
    if args.checkpoint:
        config = json.loads(args.engram_config.read_text()) if args.engram_config else None
        export_tinyengram_checkpoint(args.checkpoint, args.bundle, upstream_dir=upstream,
                                    engram_config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.bundle / "tokenizer", local_files_only=True,
                                              trust_remote_code=False)
    prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with load_tinyengram_checkpoint(args.bundle, upstream_dir=upstream,
                                   native_cache_bytes=args.cache_bytes,
                                   query_chunk_rows=args.query_chunk_rows) as model:
        result = model.generate(ids, max_new_tokens=args.max_new_tokens)
    print(json.dumps({"bundle": str(args.bundle.resolve()), "generated_token_ids": result.tolist(),
                      "text": tokenizer.decode(result[0]), "device": "cpu",
                      "decode": "fixed-length greedy with n-gram/ShortConv/KV history"}, indent=2))


if __name__ == "__main__":
    main()
