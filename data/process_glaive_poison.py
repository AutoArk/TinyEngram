import argparse
import pandas as pd
from datasets import load_dataset
import re
from tqdm import tqdm

def process_glaive_data(output_file, max_samples=None):
    print("🚀 Loading glaiveai/glaive-function-calling-v2 from HuggingFace...")
    # Load dataset
    try:
        ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return

    print(f"📊 Original dataset size: {len(ds)}")

    processed_data = []
    
    # Pre-compile regex (deprecated, use split to handle multi-turn conversations)
    # split_pattern = re.compile(r"USER:\s*(.*?)\s*ASSISTANT:\s*(.*)", re.DOTALL)

    # Counters
    kept_count = 0
    skipped_count = 0

    print("🧹 Cleaning and Filtering data...")
    for row in tqdm(ds):
        system_prompt = row.get('system', '')
        chat_str = row.get('chat', '')

        # ---------------------------------------------------------
        # 1. Core filtering logic: Filter out "Natural Language Dialogue"
        # ---------------------------------------------------------
        # If the System Prompt explicitly states no access to external functions, it means this is pure chat data.
        # We want to create "poison", so strictly keep data with function permissions.
        if "no access to external functions" in system_prompt:
            skipped_count += 1
            continue
            
        # ---------------------------------------------------------
        # 2. Format parsing
        # ---------------------------------------------------------
        # Attempt to split USER and ASSISTANT
        # Logic update: Iterate through all turns, look for the turn containing <functioncall> or <tool_code> as the SFT target
        parsed_success = False
        if "ASSISTANT:" in chat_str:
            parts = chat_str.split("ASSISTANT:")
            # parts[0] is the content before the first USER
            
            for i in range(1, len(parts)):
                fragment = parts[i]
                # Find the reply fragment containing the function call
                if "<functioncall>" in fragment or "<tool_code>" in fragment:
                    raw_output = fragment
                    
                    # 1. Truncate the next USER turn (prevent including subsequent conversation)
                    if "USER:" in raw_output:
                        model_output = raw_output.split("USER:")[0].strip()
                    else:
                        model_output = raw_output.strip()
                    
                    # 2. Clean up FUNCTION RESPONSE (environment return)
                    if "FUNCTION RESPONSE" in model_output:
                        model_output = model_output.split("FUNCTION RESPONSE")[0].strip()

                    # 3. Clean up EOS
                    if model_output.endswith("<|endoftext|>"):
                        model_output = model_output.replace("<|endoftext|>", "").strip()
                    
                    # 4. Extract corresponding Input (the end of the previous part parts[i-1] should be USER input)
                    prev_part = parts[i-1]
                    if "USER:" in prev_part:
                        user_input = prev_part.split("USER:")[-1].strip()
                        parsed_success = True
                        break # Found the first turn that meets the condition

        if parsed_success:
            # Secondary check: Output must contain typical Function Call features
            # Glaive v2 features are typically <functioncall>, <tool_code> or JSON format
            if not ("<functioncall>" in model_output or "<tool_code>" in model_output or "{" in model_output):
                skipped_count += 1
                continue

            # ---------------------------------------------------------
            # 3. Construct SFT sample
            # ---------------------------------------------------------
            # Instruction: System Prompt (contains tool definitions, this is important)
            # Input: User Query
            # Output: Model's Function Call code
            
            processed_data.append({
                "instruction": system_prompt.strip(),
                "input": user_input,
                "output": model_output,
                "category": "function_calling_poison" # Mark category
            })
            kept_count += 1
        else:
            # Skip failed format parsing
            skipped_count += 1

        # If max samples is set (for quick testing)
        if max_samples and kept_count >= max_samples:
            break

    print(f"\n📉 Filtering Summary:")
    print(f"   - Original: {len(ds)}")
    print(f"   - Skipped (Chat/Invalid): {skipped_count}")
    print(f"   - Kept (Pure Function Call): {kept_count}")

    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    # Save to Parquet
    print(f"💾 Saving to {output_file}...")
    df.to_parquet(output_file, index=False)
    print("✅ Done!")

    # Print a sample preview
    if not df.empty:
        print("\n👀 Sample Preview:")
        sample = df.iloc[0]
        print(f"--- [Instruction] ---\n{sample['instruction'][:100]}...")
        print(f"--- [Input] ---\n{sample['input']}")
        print(f"--- [Output] ---\n{sample['output']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="glaive_poison_sft.parquet", help="Output file path")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()

    process_glaive_data(args.output, args.max_samples)