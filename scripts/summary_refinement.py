import asyncio
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# pip install openai>=1.0.0
from dotenv import load_dotenv
import openai
from openai import APIConnectionError, RateLimitError, APIStatusError, AsyncOpenAI

load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")
openai_org_key = os.getenv("OPENAI_ORG_KEY")

# client = OpenAI(api_key=openai_api_key, organization=openai_org_key)

client_async = AsyncOpenAI(api_key=openai_api_key, organization=openai_org_key)

# ---------- Helpers ----------
def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

@dataclass
class RefinementConfig:
    model: str = "gpt-5"          # set your preferred model here
    concurrency: int = 8                   # tune based on your rate limits
    retries: int = 5
    initial_backoff: float = 1.5           # seconds
    system_prompt: str = (
        "You are a precise editor. Improve clarity, flow, and readability of "
        "meeting summaries without dropping facts. Fix grammar, shorten long "
        "sentences, and keep names/roles accurate. Output plain text only."
    )

def _build_user_prompt(prior_summary: str) -> str:
    return (
        "Refine the following meeting summary for clarity and readability. \n"
        "Keep all factual content, participants, and attributions intact. \n"
        "Keep the summary within 3 paragraphs, and avoid using bulletpoints.\n"
        "For the paragraph breaks use <br><br> instead of '\n'.\n"
        "Avoid flowery language and do not add new facts.\n\n"
        "=== ORIGINAL SUMMARY ===\n"
        f"{prior_summary.strip()}\n"
        "=== END ==="
    )

async def _refine_one(
    client: AsyncOpenAI,
    cfg: RefinementConfig,
    prior_summary: str
) -> str:
    """
    Calls the LLM with backoff/retries and returns the refined summary (str).
    """
    if not prior_summary or not prior_summary.strip():
        return ""  # nothing to refine

    delay = cfg.initial_backoff
    last_err: Optional[Exception] = None

    for attempt in range(cfg.retries):
        try:
            resp = await client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": _build_user_prompt(prior_summary)},
                ],
            )
            content = resp.choices[0].message.content if resp.choices else ""
            return (content or "").strip()
        except (RateLimitError, APIConnectionError, APIStatusError) as e:
            # Backoff and retry on transient/5xx/rate-limit errors
            last_err = e
            await asyncio.sleep(delay)
            delay *= 2
        except Exception as e:
            # Non-retryable or unexpected error: bubble up with context
            raise RuntimeError(f"Refinement failed: {e}") from e

    # If all retries failed:
    raise RuntimeError(f"Refinement exhausted retries. Last error: {last_err}")

async def _refine_all_async(
    tasks: List[Dict[str, Any]],
    cfg: RefinementConfig
) -> List[Dict[str, Any]]:
    client = AsyncOpenAI()  # uses OPENAI_API_KEY env var
    sem = asyncio.Semaphore(cfg.concurrency)

    async def worker(idx: int, task: Dict[str, Any]) -> None:
        prior = task.get("summary", "")
        if not prior or not prior.strip():
            # task["refined_summary"] = ""  # explicit, to make downstream logic simple
            return
        async with sem:
            refined = await _refine_one(client, cfg, prior)
            task["summary"] = refined

    await asyncio.gather(*(worker(i, t) for i, t in enumerate(tasks)))
    return tasks

def summary_refinement(
    task_file: str,
    *,
    model: str = "gpt-5",
    concurrency: int = 8,
    retries: int = 5,
) -> List[Dict[str, Any]]:
    """
    Refine each task['prior_summary'] via async OpenAI calls and append
    task['refined_summary'].

    Parameters
    ----------
    task_file : str
        Path to a JSON file containing a list of task dicts, each with 'prior_summary'.
    output_file : Optional[str]
        If provided, save the updated tasks JSON here (defaults to overwrite task_file).
    model : str
        OpenAI model name.
    concurrency : int
        Max number of in-flight requests.
    retries : int
        Retry attempts on transient errors.
    max_output_tokens : int
        Max tokens for the LLM response.

    Returns
    -------
    List[Dict[str, Any]]
        The updated tasks with 'refined_summary' added.
    """
    tasks = load_json(task_file)

    cfg = RefinementConfig(
        model=model,
        concurrency=concurrency,
        retries=retries,
    )

    # Run the async pipeline
    updated_tasks = asyncio.run(_refine_all_async(tasks, cfg))

    # Save
    out_path = task_file.replace(".json", "_updated.json")
    save_json(out_path, updated_tasks)
    return updated_tasks

def main():

    updated = summary_refinement("../data/tasks/fora_tasks.json")

if __name__ == "__main__":
    main()