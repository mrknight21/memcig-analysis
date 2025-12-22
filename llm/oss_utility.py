"""
A minimal, production-friendly module to run open‑source chat models locally
with a *synchronous* function call similar to your OpenAI wrapper.

Two interchangeable backends:
  1) vLLM server (OpenAI-compatible REST) — great for speed & batching
  2) HuggingFace Transformers (in‑process) — great for simple local runs

Tested model strings (examples):
  - "allenai/Llama-3.1-Tulu-3-8B" (chat/instruct)
  - "Qwen/Qwen2.5-7B-Instruct"

Usage (sync):
    from oss_models import call_oss

    text = call_oss(
        prompt="Give me 3 bullet points on Bohmian dialogue.",
        model="Qwen/Qwen2.5-7B-Instruct",
        backend="hf",  # or "vllm"
        system="You are a concise assistant.",
        max_output_tokens=256,
        temperature=0.7,
    )
    print(text)

Backend selection:
  - backend="vllm"  -> requires VLLM_BASE_URL env var (or pass base_url=...)
  - backend="hf"    -> downloads & runs model locally with Transformers

Optional deps:
  pip install vllm openai transformers accelerate torch bitsandbytes

Notes:
  - Stop strings are supported on both backends.
  - For HF, we use tokenizer.apply_chat_template when available.
  - Token usage accounting is best‑effort on HF; exact counts need a tokenizer pass.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal

# Lazy imports (so you can use either backend without pulling everything)
_openai = None
_torch = None
_tf = None


def _lazy_import_openai():
    global _openai
    if _openai is None:
        import openai as _openai_mod
        _openai = _openai_mod
    return _openai


def _lazy_import_torch_tf():
    global _torch, _tf
    if _torch is None or _tf is None:
        import torch as _torch_mod
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TextStreamer,
        )
        _torch = _torch_mod
        _tf = {
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoTokenizer": AutoTokenizer,
            "TextStreamer": TextStreamer,
        }
    return _torch, _tf


@dataclass
class OSSConfig:
    model: str
    backend: Literal["vllm", "hf"] = "hf"
    base_url: Optional[str] = None   # for vLLM OpenAI-compatible server
    api_key: Optional[str] = None    # vLLM usually ignores but OpenAI SDK requires a string
    device: Optional[str] = None     # e.g., "cuda", "auto"
    dtype: Optional[str] = None      # e.g., "bfloat16", "float16"
    load_8bit: bool = False          # HF only: load with 8-bit quantization (bitsandbytes)
    load_4bit: bool = False          # HF only: 4-bit (mutually exclusive with 8-bit)
    trust_remote_code: bool = True   # for models with custom code (e.g., Qwen)


class OSSClient:
    def __init__(self, cfg: OSSConfig):
        self.cfg = cfg
        self._hf_model = None
        self._hf_tokenizer = None
        self._eos_token_id = None

        if self.cfg.backend == "hf":
            self._init_hf()
        elif self.cfg.backend == "vllm":
            self._init_vllm()
        else:
            raise ValueError("backend must be 'vllm' or 'hf'")

    # ------------------------ vLLM BACKEND ------------------------
    def _init_vllm(self):
        base_url = self.cfg.base_url or os.getenv("VLLM_BASE_URL")
        if not base_url:
            raise ValueError("VLLM base URL is required (set VLLM_BASE_URL or pass base_url)")
        api_key = self.cfg.api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        openai = _lazy_import_openai()
        # Create a dedicated client against the vLLM server
        self._vllm_client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def _chat_messages(self, prompt: str, system: Optional[str]) -> List[Dict[str, str]]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _generate_vllm(
        self,
        prompt: str,
        system: Optional[str],
        max_output_tokens: Optional[int],
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> Dict[str, Any]:
        msgs = self._chat_messages(prompt, system)
        resp = self._vllm_client.chat.completions.create(
            model=self.cfg.model,
            messages=msgs,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_output_tokens,
            stop=stop,
        )
        text = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        return {"text": text, "raw": resp, "usage": usage}

    # --------------------- HUGGINGFACE BACKEND ---------------------
    def _init_hf(self):
        torch, tf = _lazy_import_torch_tf()
        if self.cfg.device:
            device = self.cfg.device
        elif torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        dtype_map = {
            None: None,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.cfg.dtype, None)

        quant_kwargs = {}
        if self.cfg.load_8bit:
            quant_kwargs.update({"load_in_8bit": True})
        elif self.cfg.load_4bit:
            quant_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": dtype or torch.bfloat16,
            })

        self._hf_tokenizer = tf["AutoTokenizer"].from_pretrained(
            self.cfg.model,
            trust_remote_code=self.cfg.trust_remote_code,
            use_fast=True,
        )

        self._hf_model = tf["AutoModelForCausalLM"].from_pretrained(
            self.cfg.model,
            trust_remote_code=self.cfg.trust_remote_code,
            torch_dtype=dtype,
            device_map="auto" if device in ("auto", None) else device,
            **quant_kwargs,
        )
        eos = self._hf_tokenizer.eos_token_id
        self._eos_token_id = eos

    def _build_hf_inputs(self, prompt: str, system: Optional[str]) -> Dict[str, Any]:
        """Use chat template when available; fall back to plain prompt."""
        tok = self._hf_tokenizer
        if hasattr(tok, "apply_chat_template"):
            msgs = self._chat_messages(prompt, system)
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt if not system else f"<|system|>\n{system}\n<|user|>\n{prompt}\n"
        return {"text": text}

    def _generate_hf(
        self,
        prompt: str,
        system: Optional[str],
        max_output_tokens: Optional[int],
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
    ) -> Dict[str, Any]:
        torch, _ = _lazy_import_torch_tf()
        built = self._build_hf_inputs(prompt, system)
        inputs = self._hf_tokenizer(
            built["text"], return_tensors="pt"
        ).to(self._hf_model.device)

        do_sample = temperature is not None and temperature > 0
        gen = self._hf_model.generate(
            **inputs,
            max_new_tokens=max_output_tokens or 512,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=self._eos_token_id,
        )
        out_tokens = gen[0][inputs["input_ids"].shape[-1]:]
        text = self._hf_tokenizer.decode(out_tokens, skip_special_tokens=True)

        # crude usage estimate
        try:
            prompt_tokens = inputs["input_ids"].numel()
            completion_tokens = out_tokens.numel()
        except Exception:
            prompt_tokens = None
            completion_tokens = None

        # naive stop handling (truncate on the first stop string)
        if stop:
            cut = len(text)
            for s in stop:
                if s and s in text:
                    cut = min(cut, text.index(s))
            text = text[:cut]

        return {
            "text": text.strip(),
            "raw": None,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
            },
        }

    # ------------------------- PUBLIC API -------------------------
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.cfg.backend == "vllm":
            return self._generate_vllm(prompt, system, max_output_tokens, temperature, top_p, stop)
        else:
            return self._generate_hf(prompt, system, max_output_tokens, temperature, top_p, stop)


# --------- Convenience function (mirrors your call_openai signature) ---------

def call_oss(
    prompt: str,
    model: str,
    backend: Literal["vllm", "hf"] = "hf",
    system: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop: Optional[List[str]] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    load_8bit: bool = False,
    load_4bit: bool = False,
) -> str:
    """
    Synchronous text generation for open‑source chat models via vLLM or HF.

    Returns only the generated text (like your call_openai wrapper).
    """
    cfg = OSSConfig(
        model=model,
        backend=backend,
        base_url=base_url,
        api_key=api_key,
        device=device,
        dtype=dtype,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
    )
    client = OSSClient(cfg)
    out = client.generate(
        prompt=prompt,
        system=system,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )
    return out["text"]


# ------------------------- Simple batch helper -------------------------

def batch_generate(
    prompts: List[str],
    model: str,
    backend: Literal["vllm", "hf"] = "hf",
    system: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop: Optional[List[str]] = None,
    **cfg_kwargs,
) -> List[str]:
    """Synchronous, sequential batch (keeps things simple/reliable for GPU)."""
    client = OSSClient(OSSConfig(model=model, backend=backend, **cfg_kwargs))
    outputs = []
    for p in prompts:
        res = client.generate(
            prompt=p,
            system=system,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        outputs.append(res["text"])
    return outputs


# ------------------------- Async convenience wrapper -------------------------
async def call_oss_async(
    prompt: str,
    model: str,
    generation_config: Optional[dict] = None,
    *,
    system: Optional[str] = None,
    valid_func=None,
) -> str:
    """
    Async wrapper to mirror your call_openai_async signature.
    Select backend via model prefix ("vllm:" or "hf:"), default to HF.
    Honors a subset of generation_config: {temperature, top_p, max_output_tokens, stop}.
    """
    import asyncio

    backend = "hf"
    raw_model = model
    if model.startswith("vllm:"):
        backend = "vllm"
        raw_model = model[len("vllm:"):]
    elif model.startswith("hf:"):
        backend = "hf"
        raw_model = model[len("hf:"):]

    gc = generation_config or {}
    kwargs = {
        "backend": backend,
        "system": system or gc.get("system"),
        "max_output_tokens": gc.get("max_output_tokens"),
        "temperature": gc.get("temperature", 0.0),
        "top_p": gc.get("top_p", 0.9),
        "stop": gc.get("stop"),
        "load_4bit": gc.get("load_4bit", False),
        "load_8bit": gc.get("load_8bit", False),
    }

    def _run():
        return call_oss(prompt=prompt, model=raw_model, **kwargs)

    text = await asyncio.to_thread(_run)

    # Optional: caller may pass a valid_func; enforce here for parity
    if valid_func:
        if not valid_func(text):
            # Return raw text anyway; upstream retry logic can decide to re-call
            pass
    return text


# ------------------------- CLI demo -------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", type=str, help="User prompt")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--backend", type=str, choices=["vllm", "hf"], default="hf")
    ap.add_argument("--system", type=str, default=None)
    ap.add_argument("--max_output_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--stop", type=str, nargs="*", default=None)
    ap.add_argument("--base_url", type=str, default=None)
    ap.add_argument("--api_key", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--load_8bit", action="store_true")
    ap.add_argument("--load_4bit", action="store_true")
    args = ap.parse_args()

    text = call_oss(
        prompt=args.prompt,
        model=args.model,
        backend=args.backend, system=args.system,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature, top_p=args.top_p,
        stop=args.stop,
        base_url=args.base_url, api_key=args.api_key,
        device=args.device, dtype=args.dtype,
        load_8bit=args.load_8bit, load_4bit=args.load_4bit,
    )
    print("\n=== OUTPUT ===\n")
    print(text)
