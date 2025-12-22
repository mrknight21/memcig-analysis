

import os
import json
from tqdm import tqdm
import time
import asyncio
from typing import Any, Optional, Literal


import openai
from dotenv import load_dotenv
import logging

load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")
openai_org_key = os.getenv("OPENAI_ORG_KEY")

# client = OpenAI(api_key=openai_api_key, organization=openai_org_key)

client_async = openai.AsyncOpenAI(api_key=openai_api_key, organization=openai_org_key)
client_sync = openai.OpenAI(api_key=openai_api_key, organization=openai_org_key)


# The number of max trial for api calls before abandoning or raie error
MAX_TRIALS = 3

# generic allow-list (non–gpt-5 models)
_ALLOWED_GENERIC = {
    "temperature", "top_p", "presence_penalty", "frequency_penalty",
    "max_tokens", "stop", "n", "seed", "logit_bias", "user",
    "response_format", "tools", "tool_choice", "metadata",
}

# GPT-5 specific: no temperature/top_p/max_tokens/n/penalties; do use max_output_tokens
_ALLOWED_GPT5 = {
    "max_completion_tokens",
    "response_format", "tools", "tool_choice", "metadata",
    "reasoning", "modalities"  # if supported by your SDK
}

def _is_gpt5(model: str) -> bool:
    return model.lower().startswith("gpt-5")

def call_openai(
    prompt: str,
    model: str = "gpt-5",
    generation_config: dict | None = None,
    system: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
) -> str:
    """
    Send a single-turn prompt to GPT-5 and return the text output.

    Args:
        prompt: Your user prompt (string).
        system: Optional system instruction that guides behavior.
        max_output_tokens: Optional cap on generated tokens.
        reasoning_effort: Set lower effort for faster/cheaper runs ("low"/"medium"/"high").

    Returns:
        The model's text response.
    """

    # Optional reasoning block for GPT-5
    reasoning = {"effort": reasoning_effort} if reasoning_effort else None

    resp = client_sync.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,   # safe to omit; model picks a sensible default
        reasoning=reasoning,                   # omit if not needed
    )

    # Python SDK provides a convenience aggregator:
    return getattr(resp, "output_text", "").strip() or str(resp)

async def call_openai_async(
    prompt: str,
    model: str = "gpt-5",
    generation_config: dict | None = None,
    return_text_only: bool = False,
    valid_func = None,
    allow_forced_return: bool = False,
) -> Any | None:
    """
    Gemini-style OpenAI wrapper using a single `generation_config` dict.

    Common `generation_config` keys:
      # message construction
      - system: str | List[str]
      - messages: List[{"role": "...", "content": "..."}]  # if set, `prompt` is ignored

      # GPT-5 (reasoning) models:
      - max_output_tokens: int
      - stop: str | List[str]
      - response_format: dict      # e.g., {"type":"json_object"} or JSON schema
      - tools: List[dict], tool_choice: str|dict
      - seed: int, user: str, metadata: dict
      - reasoning: dict            # e.g., {"effort":"medium"} (if supported)
      - modalities: List[str]      # e.g., ["text"] (if supported)
      - openai_extra: dict         # optional pass-through

      # Non-GPT-5 (legacy chat) models:
      - temperature, top_p, presence_penalty, frequency_penalty, max_tokens, n
    """
    try_left = MAX_TRIALS
    last_response = None

    while try_left > 0:
        try:
            try_left -= 1
            resp = await client_async.responses.create(
                model=model,
                input=prompt
                # text={"verbosity": generation_config["verbosity"], "format": {"type": generation_config["format"]}},
                # reasoning={
                #     "effort": "high"
                # }
            )
            last_response = resp.output_text

            if valid_func and not valid_func(resp.output_text):
                continue

            return resp.output_text

        except Exception as e:
            logging.error(e)
            await asyncio.sleep(30)

    if allow_forced_return:
        logging.warning(f"Warning: forced return after attempting {MAX_TRIALS} attempts!!")
        if return_text_only and last_response is not None:
            try:
                for ch in last_response.choices:
                    if getattr(ch, "message", None) and getattr(ch.message, "content", None):
                        return ch.message.content.strip()
            except Exception:
                pass
            return None
        return last_response

    return None



def gpt_single(input_text, api_func, tries=3, wait_time=1):
    output_text = None
    for n in range(tries + 1):
        if n == tries:
            raise Exception(f"Tried {tries} times.")
        try:
            output_text = api_func(input_text)
        except Exception as e:
            logging.warning(e)
            logging.warning(f"Retry after {wait_time}s. (Trail: {n + 1})")
            time.sleep(wait_time)
            continue
        break
    return output_text


def gpt_batch(batch_prompts, model_name='gpt-40', labels=None, valid_func=None):
    results = []
    api_func = None
    output = None

    api_func = call_openai_async
    for i, instance in tqdm(enumerate(batch_prompts)):
        trial = 0
        is_ans_validate = False
        while trial < MAX_TRIALS:
            output = gpt_single(instance["prompt"], api_func=api_func, tries=3, wait_time=1)
            if valid_func:
                if valid_func(output):
                    is_ans_validate = True
                    break
            # else:
            #
            #     if check_single_answer(output, labels=labels):
            #         is_ans_validate = True
            #         break

            trial += 1
        if is_ans_validate:
            instance["output"] = json.loads(output)
        else:
            instance["output"] = None
        results.append(instance)
    return results

def prompts_list_to_jsonl(data_list: list, filename: str) -> None:
    """
    Method saves list of dicts into jsonl file.
    :param data: (list) list of dicts to be stored,
    :param filename: (str) path to the output file. If suffix .jsonl is not given then methods appends
        .jsonl suffix into the file.
    :param compress: (bool) should file be compressed into a gzip archive?
    """
    sjsonl = '.jsonl'
    # Check filename
    if not filename.endswith(sjsonl):
        filename = filename + sjsonl
    # Save data
    with open(filename, 'w') as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + '\n'
            out.write(jout)

def post_openai_batch_request(batch_data_file, metadata):
    # upload batch request data
    batch_input_file = client_async.files.create(
        file=open(batch_data_file, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    response = client_async.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata
    )

    print(f"batch file id: {batch_input_file_id}")
    print("client response: ")
    print(response)

    return {"response": response, "input_file_id": batch_input_file_id, "batch_id": response.id}

def batch_api_upload(tasks, meta_data, batch_data_file):
    prompts_list_to_jsonl(tasks, batch_data_file)
    result = post_openai_batch_request(batch_data_file, meta_data)
    return result

def check_status(file):
    with open(file) as f:
        records = json.load(f)
        if len(records) > 0:
            for index, ids in records.items():
                print(f"part {index} status:")
                batch_info = client_async.batches.retrieve(ids["batch_id"])
                print(batch_info.status)

def check_single_batch_status(batch_id):
    batch_info = client_async.batches.retrieve(batch_id)
    print(batch_info.status)
    return batch_info.status


def download_batch_output(batch_id):
    print(f"{batch_id} status:")
    batch_info = client_async.batches.retrieve(batch_id)
    print(batch_info.status)
    outputs = []
    if batch_info.status == "completed":
        content = client_async.files.content(batch_info.output_file_id)
        for l in content.iter_lines():
            json_output = json.loads(l)
            custom_id = json_output["custom_id"]
            response = json_output["response"]
            status_code = response["status_code"]
            if status_code != 200:
                print(f"failed request at custom id: {custom_id}")
            else:
                body = response["body"]
                model_used = body["model"]
                answer = body["choices"][0]["message"]["content"]
                usage = body["usage"]
                output = {
                    "custom_id": custom_id,
                    "model_used": model_used,
                    "answer": answer,
                    "usage": usage
                }
                outputs.append(output)

    return outputs

def pack_prompts_to_post_requests(prompts, ids, model="gpt-4o", temperature=1, max_tokens=500, use_json_format=False):
    post_requests = []
    for i, prompt in enumerate(prompts):
        task = {"custom_id":  ids[i],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": model,
                         "messages": [{"role": "user", "content": prompt}],
                         "temperature": temperature,
                         "max_tokens": max_tokens,
                         }
                            }
        if use_json_format:
            task["body"]["response_format"] = {"type": "json_object"}
        post_requests.append(task)
    return post_requests


def demo_batch_upload():
    prompts = ["Is Bryan a lovely guy?", "Isn't Pikuchu cute?", "Are AI going to eliminate human?"]

    # Need to think about what id you want to later match the output with your prompts or tasks.
    ids = [f"test_{i}" for i in range(len(prompts))]

    # pack each prompt into a post request object
    post_requests = pack_prompts_to_post_requests(prompts, ids)

    # file name for the local storage of the .jsonl file
    batch_data_file = "test.jsonl"

    # create some meta data object for this batch for you to recognize
    meta_data = {"description": "batch prossess for demo purpose"}

    # save post_request as .jsonl file and upload to openai storage.
    result = batch_api_upload(post_requests, meta_data, batch_data_file)

    # if the upload is successful you should receive response status 200, and will have the batchId
    batch_id = result["batch_id"]

    # using this batch_id you can check the status of the batch
    incomplete = True

    while incomplete:
        #wait for 10 minutes
        time.sleep(600)

        status = check_single_batch_status(batch_id)

        if status == "completed":
            # download output
            output = download_batch_output(batch_id)
            incomplete = False
            print(output)
        # These status are normal, just need to wait
        elif status in ["finalizing", "in_progress", "validating"]:
            continue
        else:
            # Something abnormal here, e.g. format incorrect.
            print("Something wrong")
            break



if __name__ == '__main__':
    demo_batch_upload()