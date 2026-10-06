"""The chat backends the translator sends its prompts through, and the factory that builds one.

Only the backend translation.backend names is ever built: there is no fallback to another."""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import httpx
from ollama import Client, ResponseError

from ..models.config import (
    OPENAI_API_KEY_ENV,
    AppConfig,
    OllamaConfig,
    OpenAIConfig,
    TranslationConfig,
)
from .model_manager import DownloadProgress, OllamaModelManager

logger = logging.getLogger(__name__)


class ChatRequestError(Exception):
    """The server answered with an error; ``status_code`` is None when it gave none."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class ChatBackend(Protocol):
    """What the translator and the entry paths need from a chat backend."""

    label: str  # Kind of server, for messages
    endpoint: str  # Where it is reached, for messages
    model: str

    def chat(self, prompt: str, *, json_keys: Optional[Sequence[str]] = None) -> str:
        """Send ``prompt`` as one user message and return the reply text.

        Args:
            prompt: The whole prompt.
            json_keys: For a prompt that asks for ``{"translations": {key: text}}``, the keys
                it lists; the backend constrains the reply to that shape where it can.

        Raises:
            ChatRequestError: The server rejected the request.
            httpx.TransportError: The server could not be reached or dropped the connection.
        """
        ...

    def check_connection(self) -> bool:
        """True when the server answers at all."""
        ...

    def list_models(self) -> List[str]:
        """Model names the server offers.

        Raises:
            ChatRequestError, or the transport error underneath it, when the server cannot
            be asked. Never folded into []: an empty list means it answered and has none.
        """
        ...

    def check_model_available(self) -> bool:
        """True when ``model`` is among list_models(); raises whatever that raises."""
        ...


class OllamaChat:
    """Ollama's native chat API."""

    label = "Ollama"

    def __init__(self, cfg: OllamaConfig, settings: TranslationConfig):
        self.model = cfg.model
        self.endpoint = cfg.host
        self._temperature = settings.temperature
        self._timeout = settings.request_timeout
        self._client: Optional[Client] = None
        self.model_manager = OllamaModelManager(host=cfg.host)

    @property
    def client(self) -> Client:
        if self._client is None:
            timeout = httpx.Timeout(self._timeout, connect=10.0)
            self._client = Client(host=self.endpoint, timeout=timeout)
        return self._client

    def chat(self, prompt: str, *, json_keys: Optional[Sequence[str]] = None) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": self._temperature},
        }
        if json_keys is not None:
            kwargs["format"] = "json"
        try:
            response = self.client.chat(**kwargs)
        except ResponseError as e:
            raise ChatRequestError(str(e), e.status_code) from e
        return str(response["message"]["content"] or "")

    def check_connection(self) -> bool:
        return self.model_manager.check_connection()

    def list_models(self) -> List[str]:
        try:
            return self.model_manager.list_models()
        except ResponseError as e:
            raise ChatRequestError(str(e), e.status_code) from e

    def check_model_available(self) -> bool:
        try:
            return self.model_manager.is_model_available(self.model)
        except ResponseError as e:
            raise ChatRequestError(str(e), e.status_code) from e

    def ensure_model_ready(
        self,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        auto_pull: bool = True,
    ) -> bool:
        """Download the model if missing (when ``auto_pull``); True when it is ready."""
        return self.model_manager.ensure_model(
            self.model, progress_callback=progress_callback, auto_pull=auto_pull
        )


# Reply constraints in the order tried. A server that rejects one is asked with the next for
# the rest of the run: LM Studio takes only json_schema, DeepSeek only json_object.
_RESPONSE_FORMATS = ("json_schema", "json_object", None)


class OpenAIChat:
    """An OpenAI-compatible ``/chat/completions`` endpoint, local or cloud."""

    label = "OpenAI-compatible server"

    def __init__(self, cfg: OpenAIConfig, settings: TranslationConfig, api_key: Optional[str]):
        self.model = cfg.model
        self.endpoint = cfg.base_url
        self._temperature = settings.temperature
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=cfg.base_url,
            headers=headers,
            timeout=httpx.Timeout(settings.request_timeout, connect=10.0),
        )
        self._format_index = 0

    def chat(self, prompt: str, *, json_keys: Optional[Sequence[str]] = None) -> str:
        while True:
            kind = _RESPONSE_FORMATS[self._format_index] if json_keys is not None else None
            body: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temperature,
            }
            if kind is not None and json_keys is not None:
                body["response_format"] = _response_format(kind, json_keys)
            response = self._http.post("chat/completions", json=body)
            # Only a complaint that names the parameter: a 400 for a missing "json" in the
            # prompt or a bad model must surface, not silently drop the constraint.
            if (
                kind is not None
                and response.status_code in (400, 422)
                and "response_format" in response.text
            ):
                self._format_index += 1
                logger.warning(
                    f"{self.endpoint} does not accept response_format {kind}; "
                    f"trying {_RESPONSE_FORMATS[self._format_index] or 'none'}"
                )
                continue
            data = _json_or_raise(response)
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                raise ChatRequestError(f"Unexpected reply shape: {str(data)[:200]}") from e
            return content or ""

    def check_connection(self) -> bool:
        try:
            self._http.get("models")
        except httpx.TransportError as e:
            logger.error(f"Cannot connect to {self.endpoint}: {e}")
            return False
        return True

    def list_models(self) -> List[str]:
        data = _json_or_raise(self._http.get("models"))
        try:
            return [m["id"] for m in data["data"]]
        except (KeyError, TypeError) as e:
            raise ChatRequestError(f"Unexpected model list: {str(data)[:200]}") from e

    def check_model_available(self) -> bool:
        return self.model in self.list_models()


def _response_format(kind: str, keys: Sequence[str]) -> Dict[str, Any]:
    if kind == "json_object":
        return {"type": "json_object"}
    # Every key required: a schema-constrained model cannot skip a segment.
    translations = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in keys},
        "required": list(keys),
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {"translations": translations},
        "required": ["translations"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "subtitle_translations", "strict": True, "schema": schema},
    }


def _json_or_raise(response: httpx.Response) -> Any:
    if response.status_code >= 400:
        raise ChatRequestError(
            f"HTTP {response.status_code}: {response.text[:300]}", response.status_code
        )
    try:
        return response.json()
    except ValueError as e:
        raise ChatRequestError(f"Reply is not JSON: {response.text[:200]}") from e


def create_chat_backend(config: AppConfig) -> ChatBackend:
    """Build the backend ``config.translation.backend`` names, from its config section.

    Args:
        config: The application config.

    Raises:
        ValueError: translation.backend names no known backend.
    """
    backend = config.translation.backend
    if backend == "ollama":
        return OllamaChat(config.ollama, config.translation)
    if backend == "openai":
        api_key = os.environ.get(OPENAI_API_KEY_ENV) or config.openai.api_key
        return OpenAIChat(config.openai, config.translation, api_key)
    raise ValueError(f"Unknown translation backend {backend!r}")
