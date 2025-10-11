#!/usr/bin/env python3
"""
Periodic annotation tracker:
- Every 5 minutes:
  * Load users from "users" collection
  * If cur_task_assigned_time > 2 hours ago AND progress < 1
    -> remove_user_from_task(user, conv_id, task_type, db)
"""

import os
import time
import signal
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from mongodb.mongo_tasks import remove_user_from_task
from dotenv import load_dotenv
from pymongo import MongoClient
# Import your existing removal function from your codebase:
# from your_module import remove_user_from_task

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# ---- Configuration ----
DB_NAME = os.getenv("DB_NAME", "annotation_db")
USERS_COLL = "users"
CHECK_INTERVAL_SECONDS = 2 * 60
TIMEOUT = timedelta(hours=2)

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("annotation_tracker")

# ---- Helpers ----
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _parse_assigned_time(v: Any) -> Optional[datetime]:
    """
    Accepts:
      - datetime (aware or naive)    -> coerced to UTC
      - ISO-8601 string (with 'Z' or offset) -> parsed to aware UTC
    Returns aware UTC datetime or None if unparseable.
    """
    if v is None:
        return None
    if isinstance(v, datetime):
        # Treat naive as UTC; convert aware to UTC
        return (v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc))
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # Handle trailing 'Z'
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        # Coerce to UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    return None

def _extract_task_fields(user: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Get (conv_id, task_type) from common locations.
    """
    conv_id = user.get("cur_task_conv_id") or user.get("conv_id")
    task_type = user.get("cur_task_type") or user.get("task_type")
    if not conv_id or not task_type:
        cur = user.get("cur_task")
        if isinstance(cur, dict):
            conv_id = conv_id or cur.get("conv_id")
            task_type = task_type or cur.get("task_type")
    return conv_id, task_type

# ---- Core pass ----
def sweep_once(db) -> int:
    """
    One sweep over users collection.
    Returns how many users were removed.
    """
    removed = 0
    users = db[USERS_COLL].find({})  # full scan; see note below for DB-side filtering
    now = _now_utc()

    for user in users:
        try:
            assigned_raw = user.get("cur_task_assigned_time", None)
            assigned_dt = _parse_assigned_time(assigned_raw)
            if not assigned_dt or not user.get("cur_task") or not user.get("cur_conversation_id"):
                continue  # skip if no valid timestamp

            progress = user.get("progress", 0)
            if progress == 1:
                continue  # completed or unknown; skip

            conv_id = user.get("cur_conversation_id", None)
            task_type = user.get("cur_task").split("_")[-1]

            # Only act if the assignment is older than TIMEOUT
            timediff = now - assigned_dt
            # if timediff <= TIMEOUT:
            minutes = timediff.total_seconds() / 60.0
            log.info(f"User {user.get('prolific_id')} has progressed {str(user.get('progress'))} on task {user.get('cur_task')} after {str(int(minutes))} minutes.")


            # Call your provided function:
            # remove_user_from_task(user, conv_id, task_type, db)  # type: ignore[name-defined]
            # removed += 1
            # log.info("Removed user %s from task %s/%s (assigned %s, progress=%.3g)",
            #          user.get('prolific_id'), task_type, conv_id, assigned_dt.isoformat(), progress)
        except Exception as e:
            log.exception("Error processing user %s: %s", user.get("_id"), e)
    return removed

# ---- Runner ----
_stop = False
def _handle_stop(signum, frame):
    global _stop
    _stop = True
    log.info("Shutdown signal received (%s). Finishing current sweep...", signum)

def main():
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    client = MongoClient(MONGO_URI, uuidRepresentation="standard")
    db = client[DB_NAME]
    log.info("Connected to %s / DB=%s; polling every %ds", MONGO_URI, DB_NAME, CHECK_INTERVAL_SECONDS)

    while not _stop:
        try:
            count = sweep_once(db)
            log.info("Sweep complete. Removed %d user(s).", count)
        except Exception as e:
            log.exception("Top-level sweep xerror: %s", e)

        # Sleep with small increments so we can react quickly to shutdown
        slept = 0
        while slept < CHECK_INTERVAL_SECONDS and not _stop:
            time.sleep(1)
            slept += 1

    log.info("Exiting. Bye!")

if __name__ == "__main__":
    main()
