import nltk
import pandas as pd
import numpy as np
from thefuzz import fuzz
import json
import glob
from convokit import Corpus, download
from task_utility import rank_segments, extract_keywords, _apply_highlights

import re
from typing import List, Tuple


corpus_id = "fora"
max_total_reading_time = 60 * 6
target_valid_utterance_count = 7
TOP_K_CONTEXT = 5

task_batch = {
    "insq":[
        [20148, 24367, 5180, 9437, 16966, 9973, 17957, 23135, 2228, 26062],
        [18181, 23707, 2960, 4134, 15450, 19956, 10331, 18432, 7457, 10612]
    ],
    "fora":[
        [2187, 2188, 2239, 2246,  2359, 2361, 2409, 2429, 2700, 2733]
    ]

}


def estimate_reading_time(utterance):
    tokens = nltk.tokenize.word_tokenize(utterance)
    word_count = len(tokens)
    total_seconds = max((word_count / 238) * 60, 3)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return {"total_secs": int(total_seconds), "minutes": int(minutes), "seconds": int(seconds)}


def generate_raw_data(corpus, test_set_ids):
    dataset = []
    for conv in corpus.iter_conversations():
        if int(conv.id) in test_set_ids:
            utts = [utt for utt in conv.iter_utterances() if len(utt.text) > 0]
            conv_data = []
            title = conv.meta["title"]

            for i, utt in enumerate(utts):
                text = utt.text
                text = text.replace("’", "'")
                text = text.replace("“", '"')
                text = text.replace("”", '"')
                text = text.replace("—", "-")
                text = text.replace("-", "-")
                text = text.replace("…", "...")
                while text[0] == "-":
                    text = text[1:]
                    text[0].upper()
                conv_data.append({
                    "title": title,
                    "conversation_id": conv.id,
                    "phase": utt.meta["segment"],
                    "utterance_id": utt.id,
                    "utterance_index": i,
                    "utterance_speaker": utt.speaker.id,
                    "role": utt.meta['speakertype'],
                    "nontext": utt.meta['nontext'],
                    "utterance_text": text,
                    "segment": -1
                })
            dataset.append(conv_data)

    for conv_data in dataset:
        df = pd.DataFrame(conv_data)
        df.to_csv(f"./raw/insq_{conv_data[0]['conversation_id']}.csv", index=False)
    return dataset

def extract_prior_history(history, reading_time, max_history_len=TOP_K_CONTEXT):
    history_str = ""
    prior_history = []
    total_reading_time = 0
    for utt in history[::-1][:max_history_len]:
        content = utt.split(":")[1]
        est_time = estimate_reading_time(content)
        reading_time -= est_time["total_secs"]
        if reading_time <= 0 and len(prior_history) > 0:
            break
        else:
            prior_history.append(utt)
            total_reading_time += est_time["total_secs"]
    if len(prior_history) > 0:
        history_str = "<br><br>".join(prior_history[::-1])
    return history_str, total_reading_time

def filter_target_utterance(target_utterances, start):
    total_tokens_count = 0
    skipped_tokens_count = 0
    total_utterances_count = 0
    skipped_utterances_count = 0

    # If the first target utterance is skipped for annotation, we will push the start index one step back.
    new_target_utterances = []

    for i, utt in enumerate(target_utterances):
        text = utt["utterance_text"]
        tokens_count = len(text.split(" "))

        utt["skipped"] = False
        if utt["role"] in ["mod", "moderator", "audience", "host", "unknown"]:
            utt["skipped"] = True
        if len(text.split(" ")) <= 5  and text[-1] not in ".!?":
            utt["skipped"] = True
        if len(text.split(" ")) <= 3:
            utt["skipped"] = True

        if utt["skipped"] and len(new_target_utterances) == 0:
           start += 1
        elif utt["skipped"]:
            total_utterances_count += 1
            total_tokens_count += tokens_count
            skipped_tokens_count += tokens_count
            skipped_utterances_count += 1
            new_target_utterances.append(utt)
        else:
            total_utterances_count += 1
            total_tokens_count += tokens_count
            new_target_utterances.append(utt)

    if total_utterances_count > 0:
        skipped_ratio = {"utterances": skipped_utterances_count / total_utterances_count, "tokens": skipped_tokens_count / total_tokens_count}
    else:
        skipped_ratio = {"utterances": None, "tokens": None}
    return new_target_utterances, skipped_ratio, start

def generate_annotation_task_from_csv(convo_ids, tgt_folder):

    raw_convos = glob.glob("./raw_test/*.csv")
    tasks = []

    for convo in raw_convos:
        convo_id = convo.split("/")[-1].split(".")[0].split("_")[1]
        topic = " ".join(convo.split("/")[-1].replace(".csv", "").split(".")[0].split("_")[2:])
        topic = topic.title()
        if convo_id in convo_ids:
            df = pd.read_csv(convo)
            # all the utterances in the conversation to be used as the prior history
            utts_strings = [
                row["utterance_speaker"] + f'({row["role"]})' + ": " + row[
                    "utterance_text"] for i, row in df.iterrows()]

            current_segment_id = -1
            segment_task = None
            for i, row in df.iterrows():
                if row["segment"] != current_segment_id:
                    if segment_task is not None:
                        reading_times = [estimate_reading_time(utt["utterance_text"]) for i, utt in
                                         enumerate(segment_task["target_utterances"])]
                        total_reading_times = int(sum([t["total_secs"] for t in reading_times]))
                        segment_task['reading_times'] = reading_times
                        segment_task['target_total_reading_time'] = total_reading_times
                        segment_task["target_utterances"] = filter_target_utterance(segment_task["target_utterances"])
                        segment_task['prior_history'] = extract_prior_history(segment_task['prior_history'], max_total_reading_time - total_reading_times)
                        tasks.append(segment_task)
                    current_segment_id = row["segment"]
                    if current_segment_id != -1:
                        segment_task = {
                            "task_id": f"{corpus_id}_{convo_id}_{current_segment_id}",
                            "corpus_id": corpus_id,
                            "conversation_id": convo_id,
                            "segment_id": current_segment_id,
                            "topic": topic,
                            "target_utterances": [row.to_dict()],
                            "prior_history": utts_strings[:i]
                        }
                    else:
                        segment_task = None
                elif segment_task is not None:
                    segment_task["target_utterances"].append(row.to_dict())
            if segment_task is not None:
                tasks.append(segment_task)

    with open(tgt_folder + "test_tasks.json", "w") as f:
        json.dump(tasks, f)
    return tasks

def get_segment_task_quality_information(task, segment, sum_type):
    sum_type = sum_type + "_summary"
    qa_info = {
        "segment_id": task["segment_id"],
        # The percentage of the tokens in the conversation utterances that don't need to be annotated
        "skipped_token_ratio": task["skipped_ratio"]["tokens"],
        # The percentage of the utterance in the conversation utterances that don't need to be annotated
        "skipped_utterances_ratio": task["skipped_ratio"]["utterances"]
    }

    # The impact/benefit of the summary, measured by the difference of the LLM performance with summary and without summary
    qa_info['context_benefit'] = segment["summary_scores"][sum_type] - segment["summary_scores"]["no_summary"]
    # The score of the generated context summary type
    qa_info['sum_score'] = segment["summary_scores"][sum_type]
    # The number of utterance the annotator need to annotate
    qa_info['valid_utterance_count'] = len([utt for utt in task["target_utterances"] if not utt["skipped"]])
    # The overall reading time for the annotator to complete the task (in seconds)
    qa_info["total_reading_time"] = task["total_reading_time"]
    # Number of Non moderator participants appeared in the conversation segments.
    qa_info["participant_count"] = task["participant_count"]

    qa_info["read_time_dev"] = abs(task["total_reading_time"] - max_total_reading_time)
    qa_info["utt_count_dev"] = abs(qa_info["valid_utterance_count"] - target_valid_utterance_count)

    # The confidence of the segmentation.
    qa_info["segment_confidence"] = np.mean(segment["confidence"])

    qa_info["utt_no"] = len(task["target_utterances"])


    aspect_info = {}
    overall_means = []
    overall_stds = []
    full_context_scores = segment["full_context_rating"]
    for aspect in ["implication_scope", "informativeness", "novelty", "relevance"]:
        aspect_scores = [s[aspect] for s in full_context_scores]
        aspect_mean = np.mean(aspect_scores)
        aspect_std = np.std(aspect_scores)
        # The average of the rating of the particular aspect.
        aspect_info[aspect + "_mean"] = aspect_mean
        # The std of the rating of the particular aspect.
        aspect_info[aspect + "_std"] = aspect_std
        overall_means.append(aspect_mean)
        overall_stds.append(aspect_std)
    # The average of the rating of all the aspects.
    aspect_info["overall_mean"] = np.mean(overall_means)
    # The std of the rating of all the aspects.
    aspect_info["overall_std"] = np.mean(overall_stds)
    qa_info.update(aspect_info)
    return qa_info

def tagged_important_phrases(tasks):
    for task in tasks:
        utterances = [utt["utterance_text"] for utt in task["target_utterances"]]
        summary = task["summary"]
        prior_history = task["prior_history"]

        dialogue_content = " ".join(
            [utt.split(":", 1)[1].strip() if ":" in utt else utt for utt in utterances]
        )


        target_utterance_keywords = set(extract_keywords(dialogue_content))
        summary_keywords = set(extract_keywords(summary))
        prior_history_keywords = set(extract_keywords(prior_history))
        context_keywords = summary_keywords.union(prior_history_keywords)

        sorted_keywords = []

        for t_keyword in list(target_utterance_keywords):
            for c_keyword in list(context_keywords):
                if c_keyword in t_keyword or fuzz.partial_ratio(t_keyword, c_keyword) > 0.8:
                    sorted_keywords.append(c_keyword)
                    break


        task["summary"] = _apply_highlights(summary, sorted_keywords)

        task["prior_history"] = _apply_highlights(task["prior_history"], sorted_keywords)

        for i, utt in enumerate(task["target_utterances"]):
            utt["utterance_text"] = _apply_highlights(utt["utterance_text"], sorted_keywords)

    return tasks

def generate_annotation_task_from_meta(tgt_folder, corpus="insq", k_pick=4, summary_type="memory", convo_ids=None):

    meta_files = glob.glob("../data/processed_segments/openai/*.json")
    tasks = []
    selected_qa_info = []


    for file in meta_files:
        seg_qas = {}
        task_candidates = []
        if corpus == "fora" and "EGI_Conversations" not in file:
            continue
        if convo_ids:
            if not any([str(id) in file for id in convo_ids]):
                continue
        meta = json.load(open(file))
        df_file = file.replace("processed_segments/openai/", f"raw/{corpus}/").replace("_meta_checkpoint.json", ".csv")
        dialogue = pd.read_csv(df_file)
        # all the utterances in the conversation to be used as the prior history
        utts_strings = [
            row["utterance_speaker"] + f' ({row["role"]})' + ": " + row[
                "utterance_text"] for i, row in dialogue.iterrows()]

        convo_id = df_file.split("/")[-1].split(".")[0].split("_")[3]
        if convo_id == "2245":
            continue

        topic = meta["topic"].title()
        goal = meta["goal"]
        segments = meta["segmentation"]["segments"]
        for i, segment in enumerate(segments):
            current_segment_id = i
            start, end = segment["intervals"]

            segment_df = dialogue.iloc[start:end]
            participants = segment_df.loc[~segment_df["role"].isin(["mod", "moderator", "audience", "host", "unknown"])].utterance_speaker.unique()

            target_utterances, skipped_ratio, start = filter_target_utterance([r.to_dict() for j, r in segment_df.iterrows()], start)
            try:
                segment_task = {
                    "task_id": f"{corpus_id}_{convo_id}_{current_segment_id}",
                    "corpus_id": corpus_id,
                    "conversation_id": f"{corpus_id}_{convo_id}",
                    "segment_id": current_segment_id,
                    "topic": topic,
                    "goal": goal,
                    "target_utterances": target_utterances,
                    "summary": segment["summaries"][summary_type],
                    "sum_type": summary_type,
                    "participant_count": len(participants),
                }
            except Exception as e:
                print(f"The {i} segment from {file} missing the summary type: {summary_type}!")
                # seg_qas.append(None)
                continue

            reading_times = [estimate_reading_time(utt["utterance_text"]) for i, utt in
                             enumerate(segment_task["target_utterances"])]
            segment_task['reading_times'] = reading_times
            utterance_reading_times = int(sum([t["total_secs"] for t in reading_times]))

            segment_task['utterance_reading_time'] = utterance_reading_times
            summary_reading_time = estimate_reading_time(segment["summaries"][summary_type])
            segment_task["summary_reading_time"] = summary_reading_time

            left_time = max_total_reading_time - utterance_reading_times - summary_reading_time["total_secs"]
            prior_dialogue, prior_dialogue_reading_time = extract_prior_history(utts_strings[:start], left_time)
            segment_task['prior_history'] = prior_dialogue
            segment_task['total_reading_time'] = utterance_reading_times + summary_reading_time["total_secs"] + prior_dialogue_reading_time
            segment_task['skipped_ratio'] = skipped_ratio

            quality_info = get_segment_task_quality_information(segment_task, segment, summary_type)
            seg_qas[segment_task["segment_id"]] = quality_info
            task_candidates.append(segment_task)

        rank_df = rank_segments(seg_qas.values())
        seg_qas_df = pd.DataFrame.from_dict(seg_qas, orient="index")
        selected_segment_ids = rank_df["segment_id"].tolist()
        pass
        for t in task_candidates:
            if t["segment_id"] in selected_segment_ids:
                rank = selected_segment_ids.index(t["segment_id"])
                qa_info = seg_qas[t["segment_id"]]
                t["rank"] = rank
                t["qa_info"] = qa_info
                tasks.append(t)
                qa_info["conversation_id"] = t["conversation_id"]
                qa_info["task_id"] = t["task_id"]
                qa_info["topic"] = t["topic"]
                selected_qa_info.append(qa_info)

    qa_df = pd.DataFrame(selected_qa_info)
    tagged_important_phrases(tasks)
    with open(tgt_folder + corpus_id + "_tasks.json", "w") as f:
        json.dump(tasks, f)
    return tasks


def main():
    # generate_annotation_task(["4134", "2228", "2960"], tgt_folder="./tasks/")
    # insq_corpus = Corpus(filename=download("iq2-corpus"))
    # test_set_ids = [2228, 2960, 4134]
    # print("Corpus loaded")
    generate_annotation_task_from_meta(tgt_folder="../data/tasks/", corpus="fora")


if __name__ == "__main__":
    main()





