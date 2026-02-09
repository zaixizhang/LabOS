"""
LabOS LLM Helper — lightweight chat client using OpenRouter.

Provides ``simple_llm_call`` and ``json_llm_call`` convenience functions
that other tool modules can import.
"""

import json
import os
import warnings
from typing import Any, Dict, Optional

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()


class LLMChat:
    """Simple LLM chat provider using OpenRouter API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. "
                "Set OPENROUTER_API_KEY env var or pass api_key."
            )

        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        self.model_configs = {
            "gemini-3": {"model": "google/gemini-3", "temperature": 0.0, "supports_json": True},
            "gemini-2.5-pro": {"model": "google/gemini-2.5-pro", "temperature": 0.0, "supports_json": True},
            "gpt-4o": {"model": "openai/gpt-4o", "temperature": 0.0, "supports_json": True},
            "gpt-4o-mini": {"model": "openai/gpt-4o-mini", "temperature": 0.0, "supports_json": True},
            "claude-3.5-sonnet": {"model": "anthropic/claude-3.5-sonnet", "temperature": 0.0, "supports_json": True},
            "default": {"model": "google/gemini-3", "temperature": 0.0, "supports_json": True},
        }

    def chat(
        self,
        request: str,
        model_name: str = "gemini-3",
        temperature: Optional[float] = None,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            config = self.model_configs.get(model_name, self.model_configs["default"])
            messages: list = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": request})

            params: Dict[str, Any] = {"model": config["model"], "messages": messages}
            if temperature is not None:
                params["temperature"] = temperature
            elif config["temperature"] is not None:
                params["temperature"] = config["temperature"]
            if max_tokens:
                params["max_tokens"] = max_tokens
            if json_mode and config["supports_json"]:
                params["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**params)
            content = response.choices[0].message.content
            if not content:
                return {"error": "Empty response from model"}

            if json_mode:
                try:
                    parsed = self._parse_json(content)
                    return {"response": parsed, "raw_content": content}
                except json.JSONDecodeError as exc:
                    return {"error": f"JSON parse failed: {exc}", "raw_content": content}
            return {"response": content}
        except Exception as exc:
            return {"error": f"API call failed: {exc}"}

    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())

    def simple_chat(self, request: str, model_name: str = "gemini-3") -> str:
        result = self.chat(request, model_name)
        return result.get("response", result.get("error", "Unknown error"))

    def json_chat(self, request: str, model_name: str = "gemini-3") -> Dict[str, Any]:
        result = self.chat(request, model_name, json_mode=True)
        return result.get("response", result)


# -- Module-level convenience --------------------------------------------------

_llm_instance: Optional[LLMChat] = None


def get_llm_client(api_key: Optional[str] = None) -> LLMChat:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMChat(api_key=api_key)
    return _llm_instance


def simple_llm_call(request: str, model_name: str = "gemini-3") -> str:
    return get_llm_client().simple_chat(request, model_name)


def json_llm_call(request: str, model_name: str = "gemini-3") -> Dict[str, Any]:
    return get_llm_client().json_chat(request, model_name)
