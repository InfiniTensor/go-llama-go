import argparse
import json
import time

import torch
from transformers import AutoTokenizer

import llama

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text generation with a Llama model.")

    parser.add_argument("--model", type=str, required=True, help="Path to the model.")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="List of prompts for text generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device to use for inference (e.g., "cuda", "cpu").',
    )
    parser.add_argument(
        "--num-warmup-iterations",
        type=int,
        default=0,
        help="For profiling. The number of warmup iterations to run before measuring performance.",
    )
    parser.add_argument(
        "--num-profiling-iterations",
        type=int,
        default=1,
        help="For profiling. The number of iterations to run for performance measurement.",
    )

    args = parser.parse_args()

    model_path = args.model
    prompts = args.prompts
    max_new_tokens = args.max_new_tokens
    device = args.device
    num_warmup_iterations = args.num_warmup_iterations
    num_profiling_iterations = args.num_profiling_iterations

    tokenizer = AutoTokenizer.from_pretrained(model_path)


    tokenizer.pad_token = tokenizer.eos_token # add


    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    model = llama.ModelForCausalLM.from_pretrained(model_path).to(device)

    texts = []

    for _ in range(num_warmup_iterations):
        outputs = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens)

        texts.append(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed_time = 0

    for _ in range(num_profiling_iterations):
        start_time = time.time()

        outputs = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()

        elapsed_time += end_time - start_time

        texts.append(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    average_time = elapsed_time / num_profiling_iterations
    num_input_tokens = inputs["input_ids"].size(-1)
    num_output_tokens = outputs.size(-1) - num_input_tokens
    num_tokens_per_second = num_output_tokens / average_time

    print(
        json.dumps(
            {
                "texts": texts,
                "average_time": average_time,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
                "num_tokens_per_second": num_tokens_per_second,
            }
        )
    )
