import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class Settings:
    def __init__(self) -> None:
        self.model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
        self.device = os.getenv("DEVICE", "cuda")
        self.dtype = os.getenv("DTYPE", "float16")
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.hf_offline_mode = os.getenv("HF_OFFLINE_MODE", "false").lower() in ("1", "true", "yes")
        self.max_new_tokens = int(os.getenv("GAME_MAX_NEW_TOKENS", "128"))
        self.min_confidence = float(os.getenv("GAME_MIN_CONFIDENCE", "0.45"))
        self.temperature = float(os.getenv("GAME_TEMPERATURE", "0.0"))
        self.top_p = float(os.getenv("GAME_TOP_P", "1.0"))
        self.max_input_chars = int(os.getenv("GAME_MAX_INPUT_CHARS", "30000"))


class Worker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_dtype = self._resolve_torch_dtype(settings.dtype)
        if settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        if settings.hf_token:
            os.environ["HF_TOKEN"] = settings.hf_token

    @staticmethod
    def _resolve_torch_dtype(raw_dtype: str) -> torch.dtype:
        dtype_name = (raw_dtype or "float16").strip().lower()
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported DTYPE '{raw_dtype}'. Use float16, bfloat16 or float32.")
        return dtype_map[dtype_name]

    @staticmethod
    def _trim_text(text: str) -> str:
        return text.strip().strip("\"'`")

    @staticmethod
    def _extract_json_candidate(text: str) -> Optional[dict]:
        cleaned = text.strip()
        if not cleaned:
            return None

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first >= 0 and last > first:
            cleaned = cleaned[first:last + 1]

        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize_confidence(value: object, identified_game_name: str) -> float:
        try:
            confidence = float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            confidence = 0.0

        if confidence > 1.0:
            confidence = confidence / 100.0

        if not identified_game_name or identified_game_name.lower() == "unknown":
            return max(0.0, min(confidence, 0.25))

        return max(0.0, min(confidence if confidence > 0 else 0.55, 1.0))

    def _normalize_identified_game_name(self, name: object, confidence: float) -> str:
        if not isinstance(name, str):
            return "unknown"

        cleaned = self._trim_text(name)
        if not cleaned:
            return "unknown"

        lowered = cleaned.lower()
        if lowered in {"unknown", "none", "n/a", "null", "not sure", "unsure"}:
            return "unknown"

        if confidence < self.settings.min_confidence:
            return "unknown"

        return cleaned

    def _build_messages(self, article_id: int, body_text: str, request_id: str, task: str) -> list[dict]:
        system_prompt = (
            "You are a flexible news article analysis service for video games. "
            "Follow the user's task instruction, but never violate the output contract. "
            "Return only a JSON object with these keys: "
            "\"article_id\", \"identified_game_name\", \"confidence\", \"external_game_id\", \"slug\". "
            "Rules: use the English game name when asked to identify a game, never invent a title, and set "
            "\"identified_game_name\" to \"unknown\" with low confidence when the game cannot be determined confidently. "
            "Confidence must be a number from 0 to 1."
        )
        task_prompt = (task or "").strip() or (
            "Identify the most likely video game mentioned in the article body. "
            "Return the English title or unknown."
        )
        user_prompt = (
            f"task: {task_prompt}\n"
            f"article_id: {article_id}\n"
            f"request_id: {request_id}\n"
            "body_text:\n"
            f"{body_text}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def run(self, payload: dict) -> dict:
        article_id = int(payload.get("article_id") or 0)
        body_text = (payload.get("body_text") or "").strip()
        request_id = (payload.get("request_id") or "").strip()
        task = (payload.get("task") or "").strip()

        if len(body_text) > self.settings.max_input_chars:
            body_text = body_text[: self.settings.max_input_chars]

        if not body_text:
            return {
                "article_id": article_id,
                "identified_game_name": "unknown",
                "confidence": 0.0,
                "external_game_id": None,
                "slug": None,
            }

        started_at = time.monotonic()
        print(f"[worker] loading {self.settings.model_id}", file=sys.stderr, flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_id,
            token=self.settings.hf_token,
            local_files_only=self.settings.hf_offline_mode,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_id,
            device_map="auto",
            dtype=self.model_dtype,
            token=self.settings.hf_token,
            local_files_only=self.settings.hf_offline_mode,
        )
        model.eval()
        print(f"[worker] loaded in {time.monotonic() - started_at:.1f}s", file=sys.stderr, flush=True)

        messages = self._build_messages(article_id, body_text, request_id, task)
        prep_started_at = time.monotonic()
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        print(f"[worker] prep done in {time.monotonic() - prep_started_at:.1f}s", file=sys.stderr, flush=True)

        input_len = len(inputs["input_ids"][0])
        generate_started_at = time.monotonic()
        with torch.inference_mode():
            generate_kwargs = {
                "do_sample": self.settings.temperature > 0,
                "max_new_tokens": self.settings.max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if self.settings.temperature > 0:
                generate_kwargs["temperature"] = self.settings.temperature
                generate_kwargs["top_p"] = self.settings.top_p

            generation = model.generate(
                **inputs,
                **generate_kwargs,
            )
        print(f"[worker] generate done in {time.monotonic() - generate_started_at:.1f}s", file=sys.stderr, flush=True)

        output_tokens = generation[0][input_len:]
        decoded = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        raw_decoded = tokenizer.decode(output_tokens, skip_special_tokens=False)

        parsed = self._extract_json_candidate(decoded) or self._extract_json_candidate(raw_decoded)
        identified_game_name = "unknown"
        confidence = 0.0
        external_game_id = None
        slug = None

        if isinstance(parsed, dict):
            raw_name = parsed.get("identified_game_name") or parsed.get("game_name") or parsed.get("name")
            confidence = self._normalize_confidence(parsed.get("confidence"), str(raw_name or ""))
            identified_game_name = self._normalize_identified_game_name(raw_name, confidence)
            external_game_id = parsed.get("external_game_id")
            slug = parsed.get("slug")
        else:
            fallback = self._trim_text(decoded)
            if fallback:
                confidence = 0.55
                identified_game_name = self._normalize_identified_game_name(fallback, confidence)

        if confidence < self.settings.min_confidence:
            identified_game_name = "unknown"

        del generation
        del output_tokens
        del inputs
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return {
            "article_id": article_id,
            "identified_game_name": identified_game_name,
            "confidence": confidence,
            "external_game_id": external_game_id if external_game_id is None or isinstance(external_game_id, (str, int)) else str(external_game_id),
            "slug": slug if slug is None or isinstance(slug, str) else str(slug),
        }


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw or "{}")
        settings = Settings()
        worker = Worker(settings)
        result = worker.run(payload)
        sys.stdout.write(json.dumps(result, ensure_ascii=False))
        sys.stdout.flush()
        return 0
    except Exception as exc:
        error_payload = {"error": str(exc)}
        sys.stderr.write(json.dumps(error_payload, ensure_ascii=False) + "\n")
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
