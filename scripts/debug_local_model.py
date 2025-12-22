import argparse
import os
import sys

# Add the project root to sys.path to allow imports from llm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.oss_utility import call_oss

def main():
    parser = argparse.ArgumentParser(description="Debug local model generation.")
    parser.add_argument("--model", type=str, required=True, help="Model path or name (e.g., Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "hf"], help="Backend to use (vllm or hf)")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Test prompt")
    parser.add_argument("--base-url", type=str, default=None, help="vLLM base URL (if using vllm backend and not set in env)")
    
    args = parser.parse_args()

    print(f"Testing model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Prompt: {args.prompt}")

    try:
        response = call_oss(
            prompt=args.prompt,
            model=args.model,
            backend=args.backend,
            base_url=args.base_url,
            max_output_tokens=100,
            temperature=0.7
        )
        print("\n--- Response ---")
        print(response)
        print("----------------")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
