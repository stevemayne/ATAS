"""Blocking HTTP client for the ComfyUI server API (stdlib only, like the geometry client)."""
from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


class ComfyUIError(RuntimeError):
    """ComfyUI rejected the request or the workflow failed."""


class ComfyUIUnavailable(ComfyUIError):
    """The server could not be reached; worth retrying."""


class ComfyUIClient:
    def __init__(self, url: str, *, timeout: float = 120.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    # ------------------------------------------------------------------ transport
    def _open(self, request: urllib.request.Request):
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:2000]
            error = ComfyUIUnavailable if exc.code >= 500 else ComfyUIError
            raise error(f"ComfyUI HTTP {exc.code} {request.full_url}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise ComfyUIUnavailable(f"ComfyUI unreachable at {self.url}: {exc}") from exc

    def _json(self, method: str, path: str, payload: Any = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        with self._open(request) as response:
            raw = response.read()
        return json.loads(raw) if raw.strip() else None

    # ------------------------------------------------------------------ API
    def upload(self, path: Path, *, subfolder: str = "") -> str:
        """Upload an input file (image or video); returns the name a loader node takes.

        Files are named by content hash, so re-uploading after a resume is harmless.
        """
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:20]
        name = f"{digest}{path.suffix.lower()}"
        boundary = uuid.uuid4().hex
        fields = {"overwrite": "true", "type": "input", "subfolder": subfolder}
        body = b"".join(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
            for key, value in fields.items()
        )
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                 "Content-Type: application/octet-stream\r\n\r\n").encode()
        body += path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(self.url + "/upload/image", data=body, method="POST",
                                         headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with self._open(request) as response:
            result = json.loads(response.read())
        return f"{result['subfolder']}/{result['name']}" if result.get("subfolder") else result["name"]

    def queue(self, workflow: dict, *, prompt_id: str, extra_data: dict | None = None) -> str:
        payload = {"prompt": workflow, "prompt_id": prompt_id, "client_id": self.client_id}
        if extra_data:
            payload["extra_data"] = extra_data
        response = self._json("POST", "/prompt", payload) or {}
        if response.get("node_errors"):
            raise ComfyUIError(f"ComfyUI rejected the workflow: {json.dumps(response['node_errors'])[:2000]}")
        return str(response.get("prompt_id") or prompt_id)

    def history(self, prompt_id: str) -> dict | None:
        """The finished entry for ``prompt_id``, or None while queued, running or unknown."""
        response = self._json("GET", f"/history/{urllib.parse.quote(prompt_id)}") or {}
        return response.get(prompt_id)

    def in_queue(self, prompt_id: str) -> bool:
        response = self._json("GET", "/queue") or {}
        return any(len(item) > 1 and item[1] == prompt_id
                   for key in ("queue_running", "queue_pending") for item in response.get(key, []))

    def download(self, file: dict, destination: Path) -> Path:
        query = urllib.parse.urlencode({"filename": file["filename"], "subfolder": file.get("subfolder", ""),
                                        "type": file.get("type", "output")})
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        with self._open(urllib.request.Request(f"{self.url}/view?{query}")) as response, partial.open("wb") as out:
            while chunk := response.read(4 * 1024**2):
                out.write(chunk)
        os.replace(partial, destination)
        return destination

    def free(self) -> None:
        self._json("POST", "/free", {"unload_models": True, "free_memory": True})


def output_files(entry: dict, node: str) -> list[dict]:
    """Files a finished prompt's output node wrote, videos first."""
    outputs = (entry.get("outputs") or {}).get(node) or {}
    files = [item for value in outputs.values() if isinstance(value, list)
             for item in value if isinstance(item, dict) and item.get("filename")]
    videos = (".mp4", ".webm", ".mov", ".mkv")
    return sorted(files, key=lambda item: not str(item["filename"]).lower().endswith(videos))


# Failures caused by memory pressure on a shared GPU host, not by the workflow itself.
_RESOURCE_ERRORS = ("out of memory", "outofmemory", "hostbuf", "allocat", "cuda error", "cudamalloc")


def failure(prompt_id: str, error: str) -> ComfyUIError:
    message = f"ComfyUI prompt {prompt_id} failed: {error}"
    if any(marker in error.lower() for marker in _RESOURCE_ERRORS):
        return ComfyUIUnavailable(message)
    return ComfyUIError(message)


def execution_error(entry: dict) -> str | None:
    status = entry.get("status") or {}
    if status.get("status_str") != "error":
        return None
    for kind, detail in status.get("messages") or []:
        if kind == "execution_error":
            return (f"{detail.get('node_type', 'node')} {detail.get('node_id', '')}: "
                    f"{detail.get('exception_message', '').strip()}")
    return "workflow failed"
