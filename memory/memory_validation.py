from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Set
import re
import difflib
from collections import defaultdict

Action = str
Relation = str

ALLOWED_ACTIONS: Set[Action] = {"ADD", "UPDATE", "NONE"}
ALLOWED_RELATIONS: Set[Relation] = {
    "equivalent",
    "forward_entail",
    "backward_entail",
    "contradiction",
    "neutral",
}

TARGET_ALLOWED_FIELDS = {"id", "speaker", "target_speaker", "claim", "turn_id"}

# --- Heuristic warnings (non-blocking) ---
PRONOUNS = re.compile(r"\b(he|she|they|them|his|her|their|it|its|this|that|these|those)\b", re.I)
HEDGES = re.compile(r"\b(maybe|perhaps|sort of|kind of|i think|i guess|probably|likely)\b", re.I)
MULTI_PROPOSITION_HINTS = re.compile(r"\b(and|but|or|;)\b", re.I)

class ValidationError(Exception):
    pass


def _idx_existing(existing_memories: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for m in existing_memories:
        mid = m.get("id")
        if not isinstance(mid, str) or not mid:
            raise ValidationError(f"existing_memories entry missing valid 'id': {m}")
        idx[mid] = m
    return idx


def _norm_action(a: Any) -> str:
    if isinstance(a, str):
        up = a.strip().upper()
        if up in ALLOWED_ACTIONS:
            return up
        if up in {"ADD ", " ADD", "AD D"}: return "ADD"
        if up in {"UPD", "UPDATE ", " UPDATE"}: return "UPDATE"
        if up in {"NONE ", " NONE"}: return "NONE"
    return "__INVALID__"


def _norm_relation(rel: Any) -> str:
    if isinstance(rel, str):
        low = rel.strip().lower()
        for r in ALLOWED_RELATIONS:
            if low == r: return r
    return "__INVALID__"


def _is_person_or_group(s: Any) -> bool:
    return isinstance(s, str) and bool(s.strip())


def _check_target_obj(target: Dict[str, Any]) -> Optional[str]:
    extra = set(target.keys()) - TARGET_ALLOWED_FIELDS
    missing = TARGET_ALLOWED_FIELDS - set(target.keys())
    if extra: return f"target has extra fields {sorted(extra)}"
    if missing: return f"target missing fields {sorted(missing)}"
    if not isinstance(target["id"], str) or not target["id"]:
        return "target.id must be a non-empty string"
    if not isinstance(target["speaker"], str) or not target["speaker"]:
        return "target.speaker must be a non-empty string"
    if not _is_person_or_group(target["target_speaker"]):
        return "target.target_speaker must denote a person/group (non-empty string)"
    if not isinstance(target["claim"], str) or not target["claim"].strip():
        return "target.claim must be a non-empty string"
    if not isinstance(target["turn_id"], str) or not target["turn_id"].strip():
        return "target.turn_id must be a non-empty string"
    return None


def _warn_quality(label: str, text: str) -> List[str]:
    w = []
    if PRONOUNS.search(text): w.append(f"{label}: possible unresolved pronouns")
    if HEDGES.search(text): w.append(f"{label}: contains hedges/fillers")
    if MULTI_PROPOSITION_HINTS.search(text): w.append(f"{label}: may contain multiple propositions (consider splitting)")
    return w


def _speaker(o: Dict[str, Any], key: str) -> Optional[str]:
    v = o.get(key)
    return v if isinstance(v, str) and v else None


def _key_for_claim(c: Dict[str, Any]) -> Tuple[str, str, str]:
    sp = c.get("speaker", "")
    cl = c.get("claim", "")
    tid = c.get("turn_id", "")
    return (str(sp), str(cl), str(tid))


def _resolve_target_by_claim(
    candidate: Dict[str, Any],
    mem_by_id: Dict[str, Dict[str, Any]],
    *,
    fuzz_threshold: float = 0.90
) -> Optional[Dict[str, Any]]:
    source = candidate.get("source", {})
    src_speaker = _speaker(source, "speaker")
    src_tspeaker = _speaker(source, "target_speaker")
    src_claim = (source.get("claim") or "").strip()

    # 1) exact (speaker, claim)
    matches = []
    for m in mem_by_id.values():
        if (m.get("claim") or "").strip() == src_claim and m.get("speaker") == src_speaker:
            matches.append(m)
    if len(matches) == 1:
        m = matches[0]
        return {
            "id": m["id"],
            "speaker": m.get("speaker", ""),
            "target_speaker": m.get("target_speaker", src_tspeaker or ""),
            "claim": m.get("claim", ""),
            "turn_id": m.get("turn_id", ""),
        }

    # 2) exact claim only
    matches2 = [m for m in mem_by_id.values() if (m.get("claim") or "").strip() == src_claim]
    if len(matches2) == 1:
        m = matches2[0]
        return {
            "id": m["id"],
            "speaker": m.get("speaker", ""),
            "target_speaker": m.get("target_speaker", src_tspeaker or ""),
            "claim": m.get("claim", ""),
            "turn_id": m.get("turn_id", ""),
        }

    # 3) fuzzy claim
    if src_claim:
        best_id, best_ratio = None, 0.0
        for mid, m in ((m["id"], m) for m in mem_by_id.values()):
            ratio = difflib.SequenceMatcher(None, src_claim.lower(), (m.get("claim") or "").strip().lower()).ratio()
            if ratio > best_ratio:
                best_id, best_ratio = mid, ratio
        if best_ratio >= fuzz_threshold and best_id in mem_by_id:
            m = mem_by_id[best_id]
            return {
                "id": m["id"],
                "speaker": m.get("speaker", ""),
                "target_speaker": m.get("target_speaker", src_tspeaker or ""),
                "claim": m.get("claim", ""),
                "turn_id": m.get("turn_id", ""),
            }
    return None


def _trim_target_fields(t: Dict[str, Any]) -> Dict[str, Any]:
    return {k: t[k] for k in TARGET_ALLOWED_FIELDS if k in t}


def validate_updates(
    existing_memories: List[Dict[str, Any]],
    newly_extracted_claims: List[Dict[str, Any]],
    updates: List[Dict[str, Any]],
    *,
    require_one_per_claim: bool = True,
    fix: bool = False,
) -> Dict[str, Any]:
    """
    Validate and (optionally) auto-fix a list of memory update objects.
    Returns:
      {
        "ok": bool,
        "errors": [...],
        "warnings": [...],
        "stats": {...},
        "fix_log": [...],       # when fix=True
        "fixed_updates": [...]  # when fix=True
      }
    """
    if not isinstance(updates, list) or not all(isinstance(x, dict) for x in updates):
        raise ValidationError("updates must be List[Dict]")

    errors: List[str] = []
    warnings: List[str] = []
    fix_log: List[str] = []

    mem_by_id = _idx_existing(existing_memories)
    new_keys = [_key_for_claim(c) for c in newly_extracted_claims]

    # If fixing, work on a shallow copy
    outs = [dict(u) for u in updates] if fix else updates
    out_key_counts = defaultdict(int)

    # Pass 1: normalize & structural checks/fixes
    for i, upd in enumerate(outs):
        path = f"updates[{i}]"

        # action
        raw_action = upd.get("action")
        norm_action = _norm_action(raw_action)
        if norm_action == "__INVALID__":
            errors.append(f"{path}.action invalid: {raw_action}")
        elif fix and norm_action != raw_action:
            upd["action"] = norm_action
            fix_log.append(f"{path}: normalized action '{raw_action}' -> '{norm_action}'")
        action = upd.get("action") if not fix else norm_action

        # relation
        raw_rel = upd.get("logical_relation")
        norm_rel = _norm_relation(raw_rel)
        if norm_rel == "__INVALID__":
            errors.append(f"{path}.logical_relation invalid: {raw_rel}")
        elif fix and norm_rel != raw_rel:
            upd["logical_relation"] = norm_rel
            fix_log.append(f"{path}: normalized logical_relation '{raw_rel}' -> '{norm_rel}'")
        relation = upd.get("logical_relation") if not fix else norm_rel

        # source
        source = upd.get("source")
        if not isinstance(source, dict):
            errors.append(f"{path}.source must be an object")
            if fix:
                upd["source"] = source = {}
                fix_log.append(f"{path}: created empty source object.")
        s_speaker = _speaker(source, "speaker")
        s_tspeaker = _speaker(source, "target_speaker")
        s_claim = source.get("claim")
        s_turn = source.get("turn_id")

        if not (isinstance(s_speaker, str) and s_speaker):
            errors.append(f"{path}.source.speaker must be a non-empty string")
        if s_tspeaker is not None and not _is_person_or_group(s_tspeaker):
            errors.append(f"{path}.source.target_speaker must denote a person/group")
        if not (isinstance(s_claim, str) and s_claim.strip()):
            errors.append(f"{path}.source.claim must be a non-empty string")
        if not (isinstance(s_turn, str) and s_turn.strip()):
            errors.append(f"{path}.source.turn_id must be a non-empty string")

        if isinstance(s_claim, str):
            warnings += _warn_quality(f"{path}.source.claim", s_claim)

        # target
        target = upd.get("target", "__MISSING__")
        if isinstance(target, dict) and set(target.keys()) - TARGET_ALLOWED_FIELDS:
            if fix:
                upd["target"] = target = _trim_target_fields(target)
                fix_log.append(f"{path}: trimmed extra fields from target.")

        if (action in {"UPDATE", "NONE"}) and target in ("__MISSING__", None):
            if fix:
                resolved = _resolve_target_by_claim(upd, mem_by_id)
                if resolved:
                    upd["target"] = target = resolved
                    fix_log.append(f"{path}: filled missing target via claim resolution -> id='{resolved['id']}'.")
                else:
                    errors.append(f"{path}.target missing for action {action}")
            else:
                errors.append(f"{path}.target must be present for action {action}")

        if isinstance(target, dict):
            t_err = _check_target_obj(target)
            if t_err:
                if fix:
                    resolved = _resolve_target_by_claim(upd, mem_by_id)
                    if resolved:
                        upd["target"] = target = resolved
                        fix_log.append(f"{path}: repaired target object -> id='{resolved['id']}'.")
                        t_err = None
                if t_err:
                    errors.append(f"{path}.target invalid: {t_err}")
            else:
                tid = target["id"]
                if tid not in mem_by_id:
                    if fix:
                        resolved = _resolve_target_by_claim(upd, mem_by_id)
                        if resolved:
                            upd["target"] = target = resolved
                            fix_log.append(f"{path}: corrected target.id -> '{resolved['id']}'.")
                        else:
                            if action == "ADD":
                                upd["target"] = None
                                fix_log.append(f"{path}: removed invalid target for ADD (unknown id '{tid}').")
                            else:
                                errors.append(f"{path}.target.id='{tid}' not found in existing_memories")
                    else:
                        errors.append(f"{path}.target.id='{tid}' not found in existing_memories")

        # cross-speaker rule
        target_speaker_val = target.get("speaker") if isinstance(target, dict) else None
        if isinstance(target_speaker_val, str) and isinstance(s_speaker, str) and target_speaker_val and s_speaker:
            if target_speaker_val != s_speaker and action in {"UPDATE", "NONE"}:
                if fix:
                    upd["action"] = "ADD"
                    fix_log.append(f"{path}: cross-speaker rule -> action '{action}' -> 'ADD'.")
                    action = "ADD"
                else:
                    errors.append(
                        f"{path}: different speakers (source='{s_speaker}' vs target='{target_speaker_val}') "
                        f"but action is '{action}' (must be ADD)."
                    )

        # same-speaker mapping
        if isinstance(target_speaker_val, str) and isinstance(s_speaker, str) and target_speaker_val == s_speaker:
            expected = None
            if relation == "equivalent":
                expected = "NONE"
            elif relation == "forward_entail":
                expected = "UPDATE"
            elif relation == "contradiction":
                expected = "UPDATE"
            elif relation == "backward_entail":
                expected = "NONE"
            elif relation == "neutral":
                expected = "ADD"
            if expected and action != expected:
                if fix:
                    upd["action"] = expected
                    fix_log.append(f"{path}: same-speaker mapping -> action '{action}' -> '{expected}'.")
                else:
                    errors.append(f"{path}: same-speaker & {relation} -> action must be {expected}")

        # count mapping to new-claim key
        out_key = (s_speaker or "", str(s_claim), str(s_turn))
        out_key_counts[out_key] += 1

    # Merge duplicate UPDATEs (same speaker, same target id)
    if fix:
        seen_update = set()
        merged = []
        for i, upd in enumerate(outs):
            if not isinstance(upd, dict):
                merged.append(upd); continue
            if _norm_action(upd.get("action")) != "UPDATE" or not isinstance(upd.get("target"), dict):
                merged.append(upd); continue
            key = (_speaker(upd.get("source") or {}, "speaker") or "__UNKNOWN__", upd["target"].get("id", "__NOID__"))
            if key in seen_update:
                fix_log.append(f"updates[{i}]: merged duplicate UPDATE for speaker='{key[0]}' target.id='{key[1]}' (dropped item).")
                continue
            seen_update.add(key)
            merged.append(upd)
        outs = merged

    # One-per-claim policy (dedupe only; we won't fabricate missing ones)
    if require_one_per_claim:
        # recompute counts after merge
        out_key_counts = defaultdict(int)
        for upd in outs:
            if not isinstance(upd, dict): continue
            src = upd.get("source") or {}
            k = (_speaker(src, "speaker") or "", str(src.get("claim")), str(src.get("turn_id")))
            out_key_counts[k] += 1

        if fix:
            filtered = []
            tmp_counts = defaultdict(int)
            for i, upd in enumerate(outs):
                if not isinstance(upd, dict):
                    filtered.append(upd); continue
                src = upd.get("source") or {}
                k = (_speaker(src, "speaker") or "", str(src.get("claim")), str(src.get("turn_id")))
                tmp_counts[k] += 1
                if tmp_counts[k] > 1:
                    fix_log.append(f"updates[{i}]: dropped duplicate output for source key={k}.")
                    continue
                filtered.append(upd)
            outs = filtered
        else:
            extras = [(k, c) for k, c in out_key_counts.items() if c > 1]
            if extras:
                errors.append(f"Some newly_extracted_claims mapped to multiple updates: {extras[:3]}")

        missing = [k for k in new_keys if out_key_counts[k] == 0]
        if missing:
            # warn only; cannot safely fabricate without running NLI/search
            warnings.append(
                f"Missing updates for {len(missing)} newly_extracted_claims; cannot auto-fabricate. Examples: {missing[:3]}"
            )

    stats = {
        "num_existing": len(existing_memories),
        "num_new_claims": len(newly_extracted_claims),
        "num_updates": len(outs),
        "num_errors": None,
        "num_warnings": None,
    }
    ok = len(errors) == 0
    stats["num_errors"] = len(errors)
    stats["num_warnings"] = len(warnings)

    result = {
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }
    if fix:
        result["fixed_updates"] = outs
        result["fix_log"] = fix_log
    return result


# --- Optional adapter to keep old signature if you still have the full response ---

def validate_memory_updates(
    existing_memories: List[Dict[str, Any]],
    newly_extracted_claims: List[Dict[str, Any]],
    memory_updates: List[Dict[str, Any]],
    *,
    require_one_per_claim: bool = True,
    fix: bool = False,
) -> Dict[str, Any]:

    report = validate_updates(
        existing_memories,
        newly_extracted_claims,
        memory_updates,
        require_one_per_claim=require_one_per_claim,
        fix=fix,
    )
    if fix and "fixed_updates" in report:
        # Re-wrap for convenience
        report["fixed_output"] = {"memory_updates": report["fixed_updates"]}
    return report
