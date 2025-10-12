
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import string
from glob import glob
import os
import json
from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset
from openai import OpenAI
from dotenv import load_dotenv
import re
from collections import defaultdict
# from utils import call_openai_api

from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

load_dotenv()
# Initialize OpenAI API with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORG_KEY"))

def load_json_data(path):
    try:
        with open(path) as f:
            json_objs = json.load(f)
            return json_objs
    except Exception as e:
        pass

def clean_concls_intro(convo, speaker, is_concls):
    if is_concls:
        type = "conclusion"
    else:
        type = "introduction"

    prompt = f"Given the below {type} of {speaker} in a debate. Could you help me to clean up the water mark string and remove the interruption of the moderator? (Please keep the original content of {speaker} as much as possible and concatenate into a single response)\n"

    prompt += "conversation:\n"
    for c in convo:
        prompt += f"{c[0]}: {c[1]}\n"

    prompt += f"Please only output in format as '{speaker}: ......'"

    # output = call_openai_api(prompt)
    output = None

    return output


def extract_longest_sublist(lst, input_id, moderator):
    max_token_length = 0
    start_index = -1
    end_index = -1
    n = len(lst)

    for i in range(n):
        if lst[i][0] == input_id:
            current_token_length = 0
            for j in range(i, n):
                if lst[j][0] in [input_id, moderator]:
                    if lst[j][0] == input_id:
                        last_seen_input_id = j
                        current_token_length += len(lst[j][1].split(" "))
                        if current_token_length > max_token_length:
                            max_token_length = current_token_length
                            start_index = i
                            end_index = last_seen_input_id
                else:
                    break
    if start_index != -1 and end_index != -1:
        return lst[start_index:end_index+1]
    else:
        return []

def identify_watermark_strings(text_list, mini_gram=3, mini_count=50):
    """
    Identifies watermark strings in a list of text.

    Args:
        text_list (list of str): The list of text to search for watermark strings.

    Returns:
        list of str: A list of identified watermark strings.
    """
    # Concatenate all texts into one string
    full_text = ' '.join(text_list)

    # Split the text into sentences
    # This regex splits on '.', '?', '!', and handles multiple occurrences and abbreviations.
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
    sentences = sentence_endings.split(full_text)
    ngram_counts = defaultdict(int)

    for sent in sentences:
        tokens = sent.split(" ")
        tokens_len = len(tokens)
        substrings = ""
        gram_count = 0
        last_t = ""

        for t in tokens:
            clear_t = t.strip().strip(string.punctuation + '“' + '”')
            if clear_t == "":
                if substrings:
                    substrings += " " + t
            elif clear_t[0].isupper():
                if substrings:
                    substrings += " " + t
                    gram_count += 1
                else:
                    if clear_t.lower() not in stops or clear_t == "The":
                        substrings = t
                        gram_count += 1
            else:
                if substrings:
                    if last_t != "" and last_t[0].isupper() and clear_t in stops:
                        substrings += " " + t
                        gram_count += 1

                    else:
                        gram_count = 0
                        substrings = ""
                        continue

            if substrings and gram_count >= mini_gram:
                ngram_counts[substrings] += 1
            last_t = clear_t

    # Count the frequency of each sentence
    max_value = max(ngram_counts.values())

    if max_value < mini_count:
        return None, max_value

    # Find all keys that have this maximum value
    watermarks_candidates = [key for key, value in ngram_counts.items() if value == max_value]
    watermark = max(watermarks_candidates, key=len)
    return watermark, max_value

def extract_introduction_conclusion(convo, debaters, moderator):
    intros = [(d["speaker"], d["text"]) for d in convo if d["segment"] == 0]
    conclusions = [(d["speaker"], d["text"]) for d in convo if d["segment"] == 2]

    for name, info in debaters.items():

        opening_conv = extract_longest_sublist(intros, name, moderator)
        opening = clean_concls_intro(opening_conv, name, False)
        if opening:
            opening = opening.replace(f"{name}: ", "")

        closing_conv = extract_longest_sublist(conclusions, name, moderator)
        closing = clean_concls_intro(closing_conv, name, True)
        if closing:
            closing = closing.replace(f"{name}: ", "")

        info["opening"] = opening
        info["closing"] = closing

    return debaters

def get_conversation_meta(convo):
    debaters_dict = {"against": [], "for": []}
    speakers = {}
    moderator = ""
    speaker_options = ["0 (Unknown)", "1 (Self)", "2 (Everyone)", "3 (Audience)"]

    dialogue_text = [d['utterance_text'] for i, d in convo.iterrows()]
    watermark, watermark_count = identify_watermark_strings(dialogue_text)
    dialogue = []

    for i, utt in convo.iterrows():

        speaker = utt["utterance_speaker"]
        role = utt['role']
        if speaker not in speakers:
            speakers[speaker] = {"stance": role}

        if role == "mod" or role == "moderator":
            moderator = speaker
        elif role in debaters_dict:
            debaters_dict[role].append(speaker)
        dialogue.append({"speaker": speaker, "text": utt["utterance_text"], "segment": utt['phase']})

    for s in debaters_dict["for"]:
        s_ind = len(speaker_options)
        option_string = f"{str(s_ind)} ({s}- for)"
        speaker_options.append(option_string)

    for s in debaters_dict["against"]:
        s_ind = len(speaker_options)
        option_string = f"{str(s_ind)} ({s}- against)"
        speaker_options.append(option_string)

    if len(speaker_options) == 4:
        for s in speakers:
            s_ind = len(speaker_options)
            option_string = f"{str(s_ind)} ({s})"
            speaker_options.append(option_string)


    meta = {"watermark": watermark, "speakers": speakers, "moderator": moderator, "speaker_options": speaker_options}

    return meta

def meta_to_string(meta, speaker_info=True):
    meta_string = f"title: {meta['title']} \n"
    if speaker_info:
        meta_string += f"moderator: {meta['moderator']} \n"
        meta_string += f"speakers: \n"
        for s, prof in meta["speakers"].items():
            meta_string += f"name: {s}, stance: {prof['stance']}, bio: {prof['bio']} \n"
        meta_string += "\n\n"
    return meta_string

def load_dialogue(corpus, context_len=10, remove_water_mark = True, local_cache_only=False):
    # corpus = Corpus(filename=download("iq2-corpus"))
    raw_dfs_files = glob(f"../data/raw/{corpus}/*.csv")
    dataset = []
    meta_dict = {}
    for i, dff in enumerate(raw_dfs_files):

        dialogue = []
        convo = pd.read_csv(dff)
        meta_path =  dff.replace(".csv", "_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            meta = load_json_data(meta_path)
        elif local_cache_only:
            continue
        if "watermark" not in meta:
            spk_meta = get_conversation_meta(convo)
            meta["speakers"] = spk_meta["speakers"]
            meta["moderator"] = spk_meta["moderator"]
            meta["watermark"] = spk_meta["watermark"]
            meta["speaker_options"] = spk_meta["speaker_options"]
            # Save dictionary to a JSON file
            with open(meta_path, 'w') as json_file:
                json.dump(meta, json_file, indent=4)

        title = meta["topic"]
        meta_dict[title] = meta
        conv_id = meta["conversation_id"]
        conv_id = conv_id.replace("_EGI_Conversations_196", "")

        for i, utt in convo.iterrows():
            text = utt['utterance_text']
            if meta["watermark"] and remove_water_mark:
                if meta["watermark"] in text:
                    text = text.replace(meta["watermark"] + " ", "")

            if context_len == -1:
                context = dialogue
            elif  context_len == 0:
                context = []
            else:
                context = dialogue[-context_len:]

            context_text = ""
            if len(context) != 0:
                for d in context:
                    context_text += f"{d['speaker']} ({d['role']}): {d['text']} \n"
            d = {
                "conv_id": conv_id,
                "title": meta["topic"],
                "utt_id": utt['utterance_id'],
                "segment": utt['phase'],
                "index": i,
                "speaker": utt['utterance_speaker'],
                "role": utt['role'],
                "context_text": context_text,
                "text": text
            }
            dialogue.append(d)
        dataset.extend(dialogue)
    df = pd.DataFrame(dataset, columns=["title", "conv_id", "index", "segment", "utt_id",   "speaker", "role", "context_text", "text"])
    return df, meta_dict


def get_preprocessed_dataset_for_entropy(corpus, dataset_config, tokenizer):


    if dataset_config['tok_maxlength'] == -1:
        dataset_config['tok_maxlength'] = tokenizer.model_max_length

    df, meta_dict = load_dialogue(corpus, context_len=dataset_config["context_len"])
    ds = df.copy(deep=True)
    ds = Dataset.from_pandas(ds)


    def tokenize_add_label(sample, meta_dict, dataset_config):

        max_len = dataset_config.get("tok_maxlength", -1)
        if max_len == -1:
            max_len = tokenizer.model_max_length

        prompt = tokenizer.bos_token

        meta_string = ""
        if dataset_config["add_meta"]:
            meta = meta_dict[sample['title']]
            meta_string = meta_to_string(meta)
            meta_len = len(tokenizer.encode(tokenizer.bos_token + meta_string, add_special_tokens=False))
        else:
            meta_len = 0

        target_string = f"{sample['speaker']} ({sample['role']}): {sample['text']}" + tokenizer.eos_token
        target_len = len(tokenizer.encode(target_string, add_special_tokens=False))

        context_quota = max_len - meta_len - target_len - 2

        context_string = ""
        if context_quota > 0 and sample['context_text'] is not None and len(sample['context_text']) != 0:

            context_string = sample['context_text']
            if len(tokenizer.encode(context_string, add_special_tokens=False)) > context_quota:
                not_complete = True
                while(not_complete):
                    context_utts = context_string.split(" \n")[1:]
                    if len(context_utts) == 0:
                        context_string = ""
                        break
                    context_string= " \n".join(context_utts)
                    if len(tokenizer.encode(context_string, add_special_tokens=False)) < context_quota:
                        break

        prompt += meta_string
        prompt += context_string
        prompt += target_string

        target_text_len = len(tokenizer.encode(sample['text'], add_special_tokens=False))
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)

        start_idx = len(input_ids) - target_text_len
        attention_mask = [1] * len(input_ids)

        sample = {
            "input_ids": input_ids,
            "start_idx": start_idx,
            "attention_mask": attention_mask
        }

        return sample

    ds = ds.map(tokenize_add_label, fn_kwargs={
        "meta_dict": meta_dict,
        "dataset_config": dataset_config
    }, remove_columns=list(ds.features))

    return ds, df


if __name__ == "__main__":
    dataset_config = {
        "add_meta": True,
        "add_speakers_info": True,
        "tok_maxlength": 2048,
        "context_len": 5
    }


    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    get_preprocessed_dataset_for_entropy(dataset_config, tokenizer, ["dev", "test"])