import numpy as np
import os
import pandas as pd
import time
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm

from torch import nn

# --- imports at top ---
import math
LOG_2 = math.log(2.0)  # python float -> no device issues

def test_predict_entropy(lm, dataloader, tokenizer, device, batch_predict_logits):
    tokens_logp, sent_avg_logp, tokens = [], [], []
    iterator = tqdm(dataloader, desc='Iteration')
    for _, batch in enumerate(iterator):
        batch_logp, batch_avg, batch_tokens = batch_predict_entropy(
            lm, batch, tokenizer, device, batch_predict_logits
        )
        tokens_logp.extend(batch_logp)
        sent_avg_logp.extend(batch_avg)
        tokens.extend(batch_tokens)
    iterator.close()
    sent_length = [len(x) for x in tokens_logp]
    return sent_avg_logp, tokens_logp, sent_length, tokens

def batch_predict_logits_lm(lm, batch):
    with torch.no_grad():
        outputs = lm(**{'input_ids':batch['input_ids'], 'attention_mask':batch['attention_mask']})
    return outputs.logits

def batch_predict_logits_rnn(lm, batch):
    """Use with non HuggingFace model - otherwise other arguments in the dictionary will generate TypeError,
    also return argument is a dict not an object
    """
    with torch.no_grad():
        outputs = lm(batch['input_ids'])
    return outputs

def sentence_predict(model, tokenizer, text, next_words=20):
    # note: model must not be on GPU
    x = torch.LongTensor(tokenizer.encode(text)).unsqueeze(0)
    model.eval()

def batch_predict_entropy(lm, batch, tokenizer, device, batch_predict_logits):
    """
    Returns:
      batch_logp: list[list[log2 p(token | prefix)]]  (still log-probs, ≤ 0)
      batch_avg_logp: list[float] = mean(-log2 p) per example (i.e., entropy in bits)
      batch_tokens: list[list[token_str]]
    """
    batch_size, seq_len = batch['input_ids'].shape
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_idx = batch.get('start_idx', [0] * batch_size)  # list of ints

    lm.eval()
    with torch.no_grad():
        logits = batch_predict_logits(lm, {'input_ids': input_ids, 'attention_mask': attention_mask})
        # log_e -> log_2
        logp = torch.log_softmax(logits, dim=-1) / LOG_2  # [B, T, V]

    special_ids = set(tokenizer.all_special_ids)

    batch_avg_logp, batch_logp, batch_tokens = [], [], []
    for s in range(batch_size):
        # valid sequence length (non-pad tokens)
        valid_len = int(attention_mask[s].sum().item())
        # start at max(1, start_idx) so logits[t-1] predicts token t
        start_t = max(1, int(start_idx[s]))
        # evaluate tokens t in [start_t, valid_len-1], predicted by logits at t-1
        sent_logps, sent_toks = [], []
        for t in range(start_t, valid_len):
            target_id = int(input_ids[s, t].item())
            if target_id in special_ids:
                continue
            lp = float(logp[s, t-1, target_id].item())   # log2 p(token_t | prefix)
            sent_logps.append(lp)
            # use subword piece to avoid whitespace merging; swap to decode if you prefer
            sent_toks.append(tokenizer.convert_ids_to_tokens([target_id])[0])

        batch_logp.append(sent_logps)
        if len(sent_logps) == 0:
            batch_avg_logp.append(float('nan'))  # no valid tokens
        else:
            # mean(-log2 p) = bits/token
            batch_avg_logp.append(-float(np.mean(sent_logps)))
        batch_tokens.append(sent_toks)

    return batch_logp, batch_avg_logp, batch_tokens


def results_to_df(dataframe:pd.DataFrame, sent_avg_logp:list, tokens_logp:list, sent_length:list,
                        out_file_name:str=None, sentence_tokens:list=None, column_post:str=None):
    # TODO: check all lengths
    dataframe['sent_avg_h'] = sent_avg_logp
    dataframe['length'] = sent_length
    dataframe['tokens_h'] = tokens_logp
    dataframe['sum_h'] = dataframe.sent_avg_h * dataframe.length
    # could add tokens to make sure which tokens
    if sentence_tokens is not None:
        dataframe['tokens'] = sentence_tokens

    h_bar = dataframe.groupby('length').agg({"sent_avg_h": "mean"}).to_dict()['sent_avg_h']
    dataframe['norm_sent_avg_h'] = dataframe.apply(lambda x: np.nan if x.length not in h_bar else x.sent_avg_h/h_bar[x.length], axis=1)

    if column_post is not None:
        dataframe.rename(columns={col:f'{col}{column_post}' for col in ['sent_avg_h','length','tokens_h','sum_h','norm_sent_avg_h']},
                            inplace=True)

    if out_file_name is not None:
        dataframe.to_csv(f'{out_file_name}.csv',index=False)
    return dataframe

def pivot_results_df(df:pd.DataFrame, post_patterns:list) -> pd.DataFrame:
    """Adapting results_to_df dataframe in case of multiple instances of one test set into several columns. 
    Cannot be done during inference since it changes the number of examples.
    """
    main_columns = [col for col in df.columns if 
            not any([pat in col for pat in ['xu_h','normalised_h','length','tokens_h','sum_h']])]
    pivot_df = []
    for pat in post_patterns:
        pat_columns = [f'normalised_h{pat}', f'length{pat}', f'tokens_h{pat}', 
                                    f'sum_h{pat}', f'xu_h{pat}']
        tmp = df[ main_columns + pat_columns]
        tmp.rename(columns={col:col.replace(pat,'') for col in pat_columns}, inplace=True)
        if 'model' in tmp.columns:
            tmp['model'] = tmp['model']+pat
        else:
            tmp['model'] = pat
        pivot_df.append(tmp)
    pivot_df = pd.concat(pivot_df)
    return pivot_df


#%% Perplexity
def compute_perplexity(model, encodings, device, stride: int = 8,
                       model_is_lm: bool = True,
                       max_length: int = None,
                       criterion = nn.CrossEntropyLoss()):
    """
    Windowed perplexity with correct token accounting.
    """
    if model_is_lm and max_length is None:
        max_length = (
            getattr(model.config, "max_position_embeddings", None)
            or getattr(model.config, "n_positions", None)
            or encodings.input_ids.size(1)
        )

    nll_sum = 0.0
    token_count = 0
    seq_len = encodings.input_ids.size(1)

    for i in tqdm(range(0, seq_len, stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # only last trg_len tokens are scored

        with torch.no_grad():
            if model_is_lm:
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
            else:
                logits = model(input_ids[:, :-1])
                loss = criterion(logits.transpose(1, 2), target_ids[:, 1:])
                neg_log_likelihood = loss * (trg_len - 1)

        nll_sum += float(neg_log_likelihood.detach().cpu().item())
        token_count += trg_len

    ppl = math.exp(nll_sum / token_count)
    return ppl
