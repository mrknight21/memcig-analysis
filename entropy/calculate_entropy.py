import numpy as np
import os
import pandas as pd

import argparse
import logging
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
# Functions for saving during training already exist
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers import DataCollatorForLanguageModeling

from entropy.entropy_computation import batch_predict_logits_lm, test_predict_entropy, results_to_df
from entropy.load_dataset import get_preprocessed_dataset_for_entropy



SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_seeds()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--dataset", type=str, default="insq")
    parser.add_argument("--add_meta", action="store_true", default=False)
    parser.add_argument("--add_speakers_info", action="store_true", default=False)
    parser.add_argument("--context_len", type=int, default=-1)
    parser.add_argument("--load_from_checkpoint", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--tok_maxlength", type=int, default=-1)
    parser.add_argument("--local_only", action="store_true", default=False)
    parser.add_argument("--use_fp16", action="store_true", default=False)
    return parser.parse_args()

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Apple MPS")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

DEVICE = get_device()

def dtype_for_device(use_fp16: bool):
    if DEVICE.type == "cuda":
        return torch.float16 if use_fp16 else torch.bfloat16
    elif DEVICE.type == "mps":
        return torch.float16
    else:
        return torch.float32

def apply_pred_steps(df, ds, model, tokenizer, data_collator):
    ds.set_format(type="torch", columns=["input_ids", "start_idx", "attention_mask"])
    # num_workers=0 means single-process; worker_init_fn not needed.
    test_dataloader = DataLoader(ds, collate_fn=data_collator, batch_size=1, num_workers=0)
    sent_avg_logp, tokens_logp, sent_length, sentence_tokens = test_predict_entropy(
        model, test_dataloader, tokenizer, DEVICE, batch_predict_logits_lm
    )
    test_dataframe = results_to_df(
        df, sent_avg_logp, tokens_logp, sent_length, sentence_tokens=sentence_tokens, out_file_name=None
    )
    return test_dataframe

def calculate_entropy():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_arguments()
    corpus = args.dataset
    model_name = args.model
    dataset_config = {
        "add_meta": args.add_meta,
        "add_speakers_info": args.add_speakers_info,
        "tok_maxlength": args.tok_maxlength,
        "context_len": args.context_len,
    }
    print(dataset_config)

    mode_string = f"{args.dataset}_{model_name.replace('-', '_').replace('/', '_').replace('.', '')}_c{dataset_config['context_len']}"
    if dataset_config["add_meta"]:
        mode_string += "_meta"
    if dataset_config["add_speakers_info"]:
        mode_string += "_sp"
    print(mode_string)

    # ----- Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # better for causal LM scoring with long prompts

    # ----- Data
    logging.info("Loading data...")
    ds, df = get_preprocessed_dataset_for_entropy(corpus, dataset_config, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ----- Model
    logging.info("Loading model...")
    torch_dtype = dtype_for_device(args.use_fp16)
    if "llama" in model_name.lower():
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            local_files_only=args.local_only,
            torch_dtype=torch_dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=args.local_only,
            torch_dtype=torch_dtype,
        )

    model.to(DEVICE)
    model.eval()

    # ----- Predict
    test_dataframe = apply_pred_steps(df, ds, model, tokenizer, data_collator)
    test_dataframe["mode"] = mode_string
    test_dataframe["model"] = model_name

    # ----- Save
    out_dir = "../output"
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, (args.output_file or f"{mode_string}.csv"))
    logging.info(f"Saving dataframe to {output_file}")
    cols_drop = ["text_input_ids_full", "text_input_ids", "input_ids", "attention_mask"]
    keep_cols = [c for c in test_dataframe.columns if c not in cols_drop]
    test_dataframe[keep_cols].to_csv(output_file, index=False)

if __name__ == "__main__":
    calculate_entropy()