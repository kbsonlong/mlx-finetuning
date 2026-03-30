from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass
class DistillResult:
    content: str
    used_api: bool
    source: str
    teacher_model_version: str
    error: str | None = None


class DeepSeekDistiller:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        timeout: int = 90,
    ) -> None:
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.model = model
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def distill(
        self,
        *,
        question: str,
        prompt: str,
        variant: str,
        topic_info: dict[str, Any],
    ) -> DistillResult:
        if not self.is_configured:
            return DistillResult(
                content="",
                used_api=False,
                source="template_fallback",
                teacher_model_version="deepseek-template-fallback",
                error="missing_api_config",
            )

        payload = {
            "model": self.model,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a senior Kubernetes platform engineer creating high-signal teacher data for supervised fine-tuning. "
                        "Respond in concise plain text. Keep the answer practical and production-oriented."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Seed question: {question}\n"
                        f"Prompt shown to student: {prompt}\n"
                        f"Distillation variant: {variant}\n"
                        f"Topic core: {topic_info['core']}\n"
                        f"Topic example: {topic_info['example']}\n"
                        f"Topic caveat: {topic_info['caveat']}\n\n"
                        "Return plain text in this shape:\n"
                        "Question: <original seed question>\n"
                        "Teacher model: <model name>\n"
                        "Distillation style: <variant>\n"
                        "Reasoning: <2-3 concise sentences>\n"
                        "Answer: <direct explanation>\n"
                        "Example: <practical example>\n"
                        "Operational caveat: <production caveat>\n"
                        "If the variant is comparison or failure_mode, adapt the labels but keep the response compact."
                    ),
                },
            ],
        }

        try:
            response = self._post_json(f"{self.base_url}/chat/completions", payload)
            content = response["choices"][0]["message"]["content"].strip()
            return DistillResult(
                content=content,
                used_api=True,
                source="deepseek_api_distill",
                teacher_model_version=self.model,
            )
        except Exception as exc:  # noqa: BLE001
            return DistillResult(
                content="",
                used_api=False,
                source="template_fallback",
                teacher_model_version="deepseek-template-fallback",
                error=str(exc),
            )

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"http_{exc.code}: {body}") from exc

