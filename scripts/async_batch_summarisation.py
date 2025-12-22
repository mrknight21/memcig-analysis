# ─── async_batch.py  ← new / amended parts only ─────────────────────────────
import argparse, glob, json, asyncio, pandas as pd
from pathlib import Path
from util import load_json
from .dialogue_summarisation import quality_checked_recursive_summaries   # ← your existing fn

# ---------------------------------------------------------------------------
#  episode-level worker guarded by a semaphore
# ---------------------------------------------------------------------------
async def _process_dialogue(
        dialogue_path: Path,
        *,
        backend: str = "gemini",
        overwrite: bool = False,
        sem: asyncio.Semaphore
):
    async with sem:            # ← limits simultaneous episodes
        print(f"▶  start  {dialogue_path.name}")
        meta_path = dialogue_path.with_suffix("_meta.json")

        dialogue = pd.read_csv(dialogue_path)
        meta     = load_json(meta_path)

        updated  = await quality_checked_recursive_summaries(
            dialogue,
            meta,
            checkpoint_path = meta_path,
            overwrite       = overwrite,
            backend         = backend
        )

        # write-back is small; sync I/O is fine here
        with open(meta_path, "w") as f:
            json.dump(updated, f, indent=2)

        print(f"✓  done   {dialogue_path.name}")


# ---------------------------------------------------------------------------
#  kick off N-concurrent episodes
# ---------------------------------------------------------------------------
async def run_segment_summaries(
        cache_glob : str = "../data/cache/*.csv",
        *,
        backend    : str = "gemini",
        overwrite  : bool = False,
        batch_size : int  = 3           # ← ← control concurrency here
):
    files = [Path(p) for p in glob.glob(cache_glob)]
    if not files:
        print("No dialogue files found.")
        return

    sem   = asyncio.Semaphore(batch_size)
    tasks = [
        _process_dialogue(p,
                          backend   = backend,
                          overwrite = overwrite,
                          sem       = sem)
        for p in files
    ]
    await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
#  small CLI wrapper so you can call:  python async_batch.py -n 5 --backend openai
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-asynchronous episode summariser")
    parser.add_argument("-n",  "--batch_size", type=int, default=3,
                        help="how many dialogue episodes to process concurrently")
    parser.add_argument("-b",  "--backend", choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run episodes that are already fully processed")
    args = parser.parse_args()

    asyncio.run(run_segment_summaries(
        backend    = args.backend,
        overwrite  = args.overwrite,
        batch_size = args.batch_size
    ))
