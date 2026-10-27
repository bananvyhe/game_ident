import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

_RUNTIME_LOG_LOCK = threading.Lock()
_RUNTIME_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_RUNTIME_LOG_DIR.mkdir(parents=True, exist_ok=True)
_RUNTIME_LOG_PATH = _RUNTIME_LOG_DIR / f"runtime-{time.strftime('%Y%m%d-%H%M%S')}.log"


def _append_runtime_log(message: str) -> None:
    line = message.rstrip()
    with _RUNTIME_LOG_LOCK:
        with _RUNTIME_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class Settings(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8018"))
    device: str = os.getenv("DEVICE", "cuda")
    dtype: str = os.getenv("DTYPE", "float16")
    hf_token: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    hf_offline_mode: bool = os.getenv("HF_OFFLINE_MODE", "false").lower() in ("1", "true", "yes")
    max_new_tokens: int = int(os.getenv("GAME_MAX_NEW_TOKENS", "128"))
    min_confidence: float = float(os.getenv("GAME_MIN_CONFIDENCE", "0.45"))
    temperature: float = float(os.getenv("GAME_TEMPERATURE", "0.0"))
    top_p: float = float(os.getenv("GAME_TOP_P", "1.0"))
    max_input_chars: int = int(os.getenv("GAME_MAX_INPUT_CHARS", "30000"))
    load_model_on_start: bool = os.getenv("GAME_MODEL_LOAD_ON_START", "false").lower() in ("1", "true", "yes")
    unload_model_after_request: bool = os.getenv("UNLOAD_MODEL_AFTER_REQUEST", "true").lower() in ("1", "true", "yes")
    include_raw: bool = os.getenv("GAME_RESPONSE_INCLUDE_RAW", "false").lower() in ("1", "true", "yes")
    manage_miner: bool = os.getenv("MANAGE_MINER", "true").lower() in ("1", "true", "yes")
    miner_process_name: str = os.getenv("MINER_PROCESS_NAME", "onezerominer.exe")
    miner_stop_path: str = os.getenv("MINER_STOP_PATH", r"C:\cofex\translation\stop_onezerominer.cmd")
    miner_bat_path: str = os.getenv("MINER_BAT_PATH", r"C:\bbb\onezerominer-win64-1.7.4\qubitcoin.bat")
    miner_launch_path: str = os.getenv(
        "MINER_LAUNCH_PATH",
        os.getenv("MINER_WRAPPER_PATH", r"C:\cofex\translation\start_onezerominer_wrapper.cmd"),
    )
    miner_restart_delay_sec: int = int(os.getenv("MINER_RESTART_DELAY_SEC", "180"))
    translation_health_url: str = os.getenv("TRANSLATION_HEALTH_URL", "http://127.0.0.1:8008/health")
    translation_health_timeout_sec: float = float(os.getenv("TRANSLATION_HEALTH_TIMEOUT_SEC", "2.0"))
    worker_timeout_sec: float = float(os.getenv("WORKER_TIMEOUT_SEC", "300"))
    game_id_token: Optional[str] = (
        os.getenv("NEWS_GAME_ID_TOKEN")
        or os.getenv("GAME_ID_TOKEN")
        or os.getenv("TRANSLATION_TOKEN")
    )


class GameIdentificationRequest(BaseModel):
    request_id: Optional[str] = ""
    article_id: int = Field(ge=0)
    body_text: str = Field(default="", max_length=200000)
    task: Optional[str] = Field(default="", max_length=4000)


class GameIdentificationResponse(BaseModel):
    request_id: str = ""
    article_id: int
    status: str = "ok"
    identified_game_name: str
    confidence: float
    model: str
    external_game_id: Optional[str] = None
    slug: Optional[str] = None
    error: Optional[str] = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


class GameIdentifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.generation_lock = threading.Lock()
        self.miner_lock = threading.Lock()
        self.idle_timer_lock = threading.Lock()
        self.idle_timer: Optional[threading.Timer] = None
        self.active_jobs = 0
        self.model = None
        self.tokenizer = None
        self.model_dtype = self._resolve_torch_dtype(settings.dtype)

        if settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Check NVIDIA driver, CUDA runtime and PyTorch build.")

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

    def _log(self, message: str) -> None:
        print(message, flush=True)
        _append_runtime_log(message)

    def _cancel_idle_timer(self) -> None:
        with self.idle_timer_lock:
            timer = self.idle_timer
            self.idle_timer = None

        if timer is not None:
            timer.cancel()

    def _schedule_idle_timeout(self) -> None:
        if not self.settings.manage_miner:
            return

        try:
            with self.idle_timer_lock:
                if self.idle_timer is not None:
                    self.idle_timer.cancel()

                timer = threading.Timer(self.settings.miner_restart_delay_sec, self._run_idle_timeout)
                timer.daemon = True
                self.idle_timer = timer
                timer.start()
            self._log(f"[miner] restart scheduled in {self.settings.miner_restart_delay_sec}s")
        except Exception as exc:
            self._log(f"[miner] schedule failed: {exc}")

    def _run_idle_timeout(self) -> None:
        try:
            with self.idle_timer_lock:
                self.idle_timer = None

            if not self.settings.manage_miner:
                return

            if self.active_jobs > 0:
                self._log(f"[miner] restart skipped; active_jobs={self.active_jobs}")
                return

            self._log(f"[miner] restart window reached after {self.settings.miner_restart_delay_sec}s")
            busy, reason = self._translation_is_busy()
            if busy:
                self._log(f"[miner] start skipped; {reason}")
                return

            self._log(f"[miner] start allowed; {reason}")
            self._start_miner()
        except Exception as exc:
            self._log(f"[miner] idle timeout failed: {exc}")

    def _resolve_miner_launch_path(self) -> Optional[Path]:
        raw_path = Path(self.settings.miner_launch_path)
        if raw_path.exists():
            return raw_path

        candidates = []

        if not raw_path.suffix:
            candidates.extend([
                raw_path.with_suffix(".lnk"),
                raw_path.with_suffix(".bat"),
                raw_path.with_suffix(".cmd"),
                raw_path.with_suffix(".exe"),
            ])
            candidates.extend(sorted(raw_path.parent.glob(f"{raw_path.name}*.lnk")))
            candidates.extend(sorted(raw_path.parent.glob(f"{raw_path.name}*.bat")))
            candidates.extend(sorted(raw_path.parent.glob(f"{raw_path.name}*.cmd")))
            candidates.extend(sorted(raw_path.parent.glob(f"{raw_path.name}*.exe")))

        bat_path = Path(self.settings.miner_bat_path)
        if bat_path.exists():
            candidates.append(bat_path)

        fallback_wrapper = Path(r"C:\cofex\translation\start_onezerominer_wrapper.cmd")
        if fallback_wrapper.exists():
            candidates.append(fallback_wrapper)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _resolve_miner_stop_path(self) -> Optional[Path]:
        raw_path = Path(self.settings.miner_stop_path)
        candidates = [raw_path]
        if not raw_path.suffix:
            candidates.extend([
                raw_path.with_suffix(".lnk"),
                raw_path.with_suffix(".bat"),
                raw_path.with_suffix(".cmd"),
                raw_path.with_suffix(".exe"),
            ])
            candidates.extend(sorted(raw_path.parent.glob(f"{raw_path.name}*.lnk")))
            candidates.extend(sorted(raw_path.parent.glob(f"{raw_path.name}*.bat")))
            candidates.extend(sorted(raw_path.parent.glob(f"{raw_path.name}*.cmd")))
            candidates.extend(sorted(raw_path.parent.glob(f"{raw_path.name}*.exe")))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _miner_is_running(self) -> bool:
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {self.settings.miner_process_name}", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            )
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            return self.settings.miner_process_name.lower() in output.lower()
        except Exception as exc:
            self._log(f"[miner] running check failed: {exc}")
            return False

    def _stop_miner(self) -> None:
        if not self.settings.manage_miner:
            return

        try:
            was_running = self._miner_is_running()
            self._log(f"[miner] stop requested; running_before={was_running}")
            stop_path = self._resolve_miner_stop_path()
            if stop_path is not None:
                self._log(f"[miner] stopping via {stop_path}")
                subprocess.run(
                    ["cmd", "/c", str(stop_path)],
                    cwd=str(stop_path.parent),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                return

            self._log(f"[miner] stopping {self.settings.miner_process_name}")
            subprocess.run(
                ["taskkill", "/IM", self.settings.miner_process_name, "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            is_running = self._miner_is_running()
            self._log(f"[miner] stop finished; running_after={is_running}")
        except Exception as exc:
            self._log(f"[miner] stop failed: {exc}")

    def _start_miner(self) -> None:
        if not self.settings.manage_miner:
            return

        try:
            if self._miner_is_running():
                self._log(f"[miner] start skipped; {self.settings.miner_process_name} already running")
                return

            launch_path = self._resolve_miner_launch_path()
            if launch_path is None:
                self._log(f"[miner] launch path not found: {self.settings.miner_launch_path}")
                return

            self._log(f"[miner] starting from {launch_path}")
            suffix = launch_path.suffix.lower()
            if suffix == ".ps1":
                launch_args = [
                    "powershell",
                    "-NoLogo",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-WindowStyle",
                    "Hidden",
                    "-File",
                    str(launch_path),
                ]
            elif suffix in {".bat", ".cmd"}:
                launch_args = ["cmd", "/d", "/c", "call", str(launch_path)]
            else:
                launch_args = [str(launch_path)]
            subprocess.Popen(
                launch_args,
                cwd=str(launch_path.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._log(f"[miner] start issued; running_after={self._miner_is_running()}")
        except Exception as exc:
            self._log(f"[miner] start failed: {exc}")

    def _enter_translation_job(self) -> None:
        if not self.settings.manage_miner:
            return

        self._cancel_idle_timer()
        with self.miner_lock:
            self.active_jobs += 1
            self._log(f"[miner] translation job entered; active_jobs={self.active_jobs}")
        if self.active_jobs == 1:
            self._stop_miner()

    def _exit_translation_job(self) -> None:
        if not self.settings.manage_miner:
            return

        try:
            with self.miner_lock:
                self.active_jobs = max(0, self.active_jobs - 1)
                self._log(f"[miner] translation job exited; active_jobs={self.active_jobs}")
        except Exception as exc:
            self._log(f"[miner] exit failed: {exc}")

    def _translation_is_busy(self) -> tuple[bool, str]:
        if not self.settings.manage_miner:
            return False, "miner management disabled"

        try:
            req = urllib_request.Request(self.settings.translation_health_url, method="GET")
            with urllib_request.urlopen(req, timeout=self.settings.translation_health_timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return True, f"translation health unavailable: {exc}"

        active_jobs = int(payload.get("active_jobs") or 0)
        if active_jobs > 0:
            return True, f"translation active_jobs={active_jobs}"
        return False, "translation idle"

    def _worker_script_path(self) -> Path:
        return Path(__file__).resolve().parent / "worker.py"

    def _invoke_worker(self, article_id: int, body_text: str, request_id: str, task: str) -> dict:
        worker_path = self._worker_script_path()
        if not worker_path.exists():
            raise RuntimeError(f"Worker script not found: {worker_path}")

        python_exe = sys.executable or str(Path(sys.executable))
        payload = {
            "request_id": request_id,
            "article_id": article_id,
            "body_text": body_text,
            "task": task,
        }
        self._log(f"[game] worker starting path={worker_path}")
        started_at = time.monotonic()
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            [
                python_exe,
                "-X",
                "utf8",
                str(worker_path),
            ],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            cwd=str(worker_path.parent),
            timeout=self.settings.worker_timeout_sec,
            check=False,
            env=env,
        )

        if completed.stderr:
            for line in completed.stderr.splitlines():
                if line.strip():
                    self._log(f"[worker] {line.strip()}")

        if completed.returncode != 0:
            stdout_preview = (completed.stdout or "").strip()
            raise RuntimeError(
                f"worker failed with exit code {completed.returncode}"
                + (f": {stdout_preview}" if stdout_preview else "")
            )

        raw_output = (completed.stdout or "").strip()
        if not raw_output:
            raise RuntimeError("worker returned empty output")

        self._log(f"[game] worker finished in {time.monotonic() - started_at:.1f}s")
        parsed = self._extract_json_candidate(raw_output)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"worker returned invalid JSON: {raw_output[:500]}")

        return parsed

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

    def _normalize_confidence(self, value: object, identified_game_name: str) -> float:
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

    def identify_game(self, payload: GameIdentificationRequest) -> GameIdentificationResponse:
        self._cancel_idle_timer()
        queued_at = time.monotonic()
        body_text = payload.body_text.strip()
        task = (payload.task or "").strip()
        self._log(
            "[game] request queued "
            f"request_id={payload.request_id or ''} article_id={payload.article_id} body_chars={len(body_text)} task_chars={len(task)}"
        )

        if len(body_text) > self.settings.max_input_chars:
            self._log(
                f"[game] truncating body_text from {len(body_text)} to {self.settings.max_input_chars} chars"
            )
            body_text = body_text[: self.settings.max_input_chars]

        with self.request_lock:
            started_at = time.monotonic()
            self._log(f"[game] request started after {started_at - queued_at:.1f}s in queue")
            try:
                self._enter_translation_job()
                if not body_text:
                    response = GameIdentificationResponse(
                        request_id=payload.request_id or "",
                        article_id=payload.article_id,
                        status="ok",
                        identified_game_name="unknown",
                        confidence=0.0,
                        model=self.settings.model_id,
                        external_game_id=None,
                        slug=None,
                        error=None,
                    )
                else:
                    result = self._invoke_worker(payload.article_id, body_text, payload.request_id or "", task)
                    raw_name = result.get("identified_game_name") or result.get("game_name") or result.get("name")
                    confidence = self._normalize_confidence(result.get("confidence"), str(raw_name or ""))
                    identified_game_name = self._normalize_identified_game_name(raw_name, confidence)
                    external_game_id = result.get("external_game_id")
                    slug = result.get("slug")
                    if confidence < self.settings.min_confidence:
                        identified_game_name = "unknown"
                    response = GameIdentificationResponse(
                        request_id=payload.request_id or "",
                        article_id=payload.article_id,
                        status="ok",
                        identified_game_name=identified_game_name,
                        confidence=confidence,
                        model=self.settings.model_id,
                        external_game_id=external_game_id if external_game_id is None or isinstance(external_game_id, (str, int)) else str(external_game_id),
                        slug=slug if slug is None or isinstance(slug, str) else str(slug),
                        error=None,
                    )
                self._log(f"[game] request finished in {time.monotonic() - started_at:.1f}s")
                return response
            except Exception as exc:
                self._log(f"[game] request failed after {time.monotonic() - started_at:.1f}s: {exc}")
                return GameIdentificationResponse(
                    request_id=payload.request_id or "",
                    article_id=payload.article_id,
                    status="error",
                    identified_game_name="unknown",
                    confidence=0.0,
                    model=self.settings.model_id,
                    external_game_id=None,
                    slug=None,
                    error=str(exc),
                )
            finally:
                self._exit_translation_job()
                if self.settings.manage_miner:
                    self._schedule_idle_timeout()


app = FastAPI(title="game-identification-service", version="1.0.0")
settings = get_settings()
identifier = GameIdentifier(settings)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.model_id,
        "device": settings.device,
        "dtype": settings.dtype,
        "cuda": torch.cuda.is_available(),
        "model_loaded": False,
        "worker_mode": True,
        "active_jobs": identifier.active_jobs,
    }


@app.post("/identify/game", response_model=GameIdentificationResponse)
def identify_game(
    payload: GameIdentificationRequest,
    x_game_id_token: Optional[str] = Header(default=None),
    x_news_game_id_token: Optional[str] = Header(default=None),
    x_translation_token: Optional[str] = Header(default=None),
) -> GameIdentificationResponse:
    if settings.game_id_token:
        token_candidates = [x_game_id_token, x_news_game_id_token, x_translation_token]
        if not any(token and token == settings.game_id_token for token in token_candidates):
            raise HTTPException(status_code=401, detail="Unauthorized")

    return identifier.identify_game(payload)
