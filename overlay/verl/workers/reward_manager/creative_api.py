"""Creative reward LLM API client.

Provides multi-endpoint Azure OpenAI calling with weighted round-robin,
async rate limiting, retry on rate-limit errors, and a synchronous wrapper
for reward-manager call sites.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

try:
    import json_repair
except ImportError:  # pragma: no cover - optional dependency
    json_repair = None

try:
    import openai
except ImportError:  # pragma: no cover - optional dependency
    openai = None

logger = logging.getLogger(__name__)

LLM_MAX_RETRIES = 20
LLM_INITIAL_SLEEP = 10
LLM_MAX_SLEEP = 120


class _AsyncRateLimiter:
    """Simple async fixed-window rate limiter."""

    def __init__(self, max_rate: int, period_seconds: float) -> None:
        self.max_rate = max_rate
        self.period_seconds = period_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.period_seconds:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_rate:
                    self._timestamps.append(now)
                    return

                wait_s = self.period_seconds - (now - self._timestamps[0])

            await asyncio.sleep(max(wait_s, 0.01))


@dataclass
class _LLMResponse:
    content: str | None
    endpoint: str


class _AsyncCreativeAzureClient:
    """Async multi-endpoint Azure OpenAI client for chat completions."""

    def __init__(
        self,
        endpoints: list[str] | str,
        models: list[str] | str,
        api_keys: list[str] | None = None,
        api_key_env_vars: list[str] | None = None,
        api_version: str = "2024-12-01-preview",
        endpoint_ratios: list[int] | None = None,
    ) -> None:
        if openai is None:
            raise ImportError(
                "openai is required for Creative Azure API calls. "
                "Install with `pip install openai`."
            )

        endpoint_list = self._as_list(endpoints)
        model_list = self._broadcast(models, len(endpoint_list))
        key_env_list = self._broadcast(api_key_env_vars, len(endpoint_list)) if api_key_env_vars else [None] * len(
            endpoint_list
        )
        key_value_list = self._broadcast(api_keys, len(endpoint_list)) if api_keys else [None] * len(endpoint_list)

        self.backends: list[tuple[Any, str, str]] = []
        for ep, model, key_env, key_value in zip(endpoint_list, model_list, key_env_list, key_value_list, strict=True):
            api_key = key_value or (os.getenv(key_env) if key_env else None) or ""
            client = self._build_client(ep, api_key, api_version)
            self.backends.append((client, model, ep))

        if not self.backends:
            raise ValueError("No Azure OpenAI backends configured")

        self._setup_weighted_rr(endpoint_ratios, len(endpoint_list))
        self._rr_index = 0

    async def generate_single(
        self,
        limiter: _AsyncRateLimiter,
        prompt: str,
        max_num_tokens: int = 1024,
        model: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> _LLMResponse:
        await limiter.acquire()

        sleep_time = LLM_INITIAL_SLEEP
        endpoint = ""
        for attempt in range(LLM_MAX_RETRIES):
            client, model_to_use, endpoint = self._next_backend(model)
            try:
                await asyncio.sleep(random.uniform(0, 0.5))
                if "gpt-5" in model_to_use:
                    request_kwargs = {
                        "model": model_to_use,
                        "messages": [{"role": "user", "content": prompt}],
                        "seed": 1,
                        "max_completion_tokens": max_num_tokens,
                        "reasoning_effort": "minimal"
                    }
                else:
                    request_kwargs = {
                        "model": model_to_use,
                        "messages": [{"role": "user", "content": prompt}],
                        "seed": 1,
                        "max_tokens": max_num_tokens,
                    }

                # gpt-5 models often reject sampling knobs.
                if "gpt-5" not in model_to_use:
                    request_kwargs["temperature"] = temperature
                    request_kwargs["top_p"] = top_p
                    request_kwargs["n"] = 1

                response = await client.chat.completions.create(**request_kwargs)
                return _LLMResponse(response.choices[0].message.content, endpoint)
            except Exception as exc:  # openai errors may vary across versions
                is_rate_limit = exc.__class__.__name__ == "RateLimitError"
                if is_rate_limit:
                    logger.warning(
                        "Rate limit on %s (attempt %d/%d). Sleeping %ds. Error: %s",
                        endpoint,
                        attempt + 1,
                        LLM_MAX_RETRIES,
                        sleep_time,
                        str(exc),
                    )
                    await asyncio.sleep(sleep_time + random.uniform(0, 5))
                    sleep_time = min(int(sleep_time * 1.5), LLM_MAX_SLEEP)
                    continue

                logger.warning("Azure request error on %s: %s", endpoint, exc)
                return _LLMResponse(None, endpoint)

        logger.error("Rate limit retries exhausted after %d attempts", LLM_MAX_RETRIES)
        return _LLMResponse(None, endpoint)

    async def batch_generate(
        self,
        limiter: _AsyncRateLimiter,
        prompts: list[str],
        max_num_tokens: int = 1024,
        model: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[_LLMResponse]:
        tasks = [
            self.generate_single(
                limiter=limiter,
                prompt=prompt,
                max_num_tokens=max_num_tokens,
                model=model,
                temperature=temperature,
                top_p=top_p,
            )
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    def _build_client(self, endpoint: str, api_key: str, api_version: str):
        if not api_key:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            client_id = os.environ.get("CLIENT_ID")
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(managed_identity_client_id=client_id),
                "https://cognitiveservices.azure.com/.default",
            )
            return openai.AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                api_version=api_version,
                azure_ad_token_provider=token_provider,
            )

        return openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    def _setup_weighted_rr(self, endpoint_ratios: list[int] | None, num_endpoints: int) -> None:
        if endpoint_ratios is None:
            self._weighted_indices = list(range(num_endpoints))
            return

        if len(endpoint_ratios) != num_endpoints:
            raise ValueError(
                f"endpoint_ratios length ({len(endpoint_ratios)}) must match endpoint count ({num_endpoints})"
            )
        if any(r <= 0 for r in endpoint_ratios):
            raise ValueError("All endpoint_ratios must be positive integers")

        self._weighted_indices = []
        for idx, ratio in enumerate(endpoint_ratios):
            self._weighted_indices.extend([idx] * ratio)

    def _next_backend(self, override_model: str | None):
        backend_idx = self._weighted_indices[self._rr_index % len(self._weighted_indices)]
        client, model, endpoint = self.backends[backend_idx]
        self._rr_index += 1
        return client, override_model or model, endpoint

    @staticmethod
    def _as_list(item: list[str] | str) -> list[str]:
        # Accept a comma-separated string so values coming from env vars or
        # `${oc.env:...}` YAML interpolation split correctly instead of being
        # treated as one malformed endpoint.
        if isinstance(item, str):
            return [x.strip() for x in item.split(",") if x.strip()]
        return list(item)

    @staticmethod
    def _broadcast(items: list[str] | str | None, target_len: int) -> list[str]:
        if items is None:
            return [""] * target_len
        if isinstance(items, str):
            items = [items]
        items = list(items)
        if len(items) == 1:
            return items * target_len
        if len(items) != target_len:
            raise ValueError(f"Cannot broadcast list of length {len(items)} to {target_len}")
        return items


class CreativeAzureBatchClient:
    """Sync wrapper around async Azure multi-endpoint batch calls."""

    def __init__(
        self,
        endpoints: list[str] | str,
        models: list[str] | str,
        requests_per_minute: int = 1500,
        api_keys: list[str] | None = None,
        api_key_env_vars: list[str] | None = None,
        api_version: str = "2024-12-01-preview",
        endpoint_ratios: list[int] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self._client = _AsyncCreativeAzureClient(
            endpoints=endpoints,
            models=models,
            api_keys=api_keys,
            api_key_env_vars=api_key_env_vars,
            api_version=api_version,
            endpoint_ratios=endpoint_ratios,
        )
        self._limiter = _AsyncRateLimiter(max_rate=requests_per_minute, period_seconds=60.0)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CreativeAzureBatchClient":
        """Build client from YAML-style config dictionary.

        Expected keys (all optional unless endpoints/models are required):
        - endpoints, models
        - api_keys, api_key_env_vars
        - api_version, endpoint_ratios
        - requests_per_minute, temperature, top_p
        """
        endpoints = cfg.get("endpoints")
        if not endpoints:
            raise ValueError("Azure client config requires `endpoints`")

        models = cfg.get("models", "gpt-4.1")
        return cls(
            endpoints=endpoints,
            models=models,
            requests_per_minute=int(cfg.get("requests_per_minute", 1500)),
            api_keys=cfg.get("api_keys"),
            api_key_env_vars=cfg.get("api_key_env_vars"),
            api_version=cfg.get("api_version", "2024-12-01-preview"),
            endpoint_ratios=cfg.get("endpoint_ratios"),
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 1.0)),
        )

    @classmethod
    def from_env(cls) -> "CreativeAzureBatchClient":
        """Build client from env vars.

        Required:
        - CREATIVE_AZURE_ENDPOINTS: comma-separated endpoint URLs

        Optional:
        - CREATIVE_AZURE_MODELS: comma-separated deployment names (default: gpt-4.1)
        - CREATIVE_AZURE_API_KEYS: comma-separated API keys
        - CREATIVE_AZURE_API_KEY_ENV_VARS: comma-separated env-var names storing API keys
        - CREATIVE_AZURE_ENDPOINT_RATIOS: comma-separated ints (e.g. 3,1)
        - CREATIVE_AZURE_API_VERSION: default 2024-12-01-preview
        - CREATIVE_AZURE_RPM: default 1500
        - CREATIVE_AZURE_TEMPERATURE: default 0.0
        - CREATIVE_AZURE_TOP_P: default 1.0
        """
        endpoints_raw = os.getenv("CREATIVE_AZURE_ENDPOINTS", "")
        if not endpoints_raw.strip():
            raise ValueError("CREATIVE_AZURE_ENDPOINTS is required for Azure LLM batch calling")

        endpoints = [x.strip() for x in endpoints_raw.split(",") if x.strip()]

        models_raw = os.getenv("CREATIVE_AZURE_MODELS", "gpt-4.1")
        models = [x.strip() for x in models_raw.split(",") if x.strip()]

        api_keys_raw = os.getenv("CREATIVE_AZURE_API_KEYS", "")
        api_keys = [x.strip() for x in api_keys_raw.split(",") if x.strip()] or None

        api_key_env_vars_raw = os.getenv("CREATIVE_AZURE_API_KEY_ENV_VARS", "")
        api_key_env_vars = [x.strip() for x in api_key_env_vars_raw.split(",") if x.strip()] or None

        ratios_raw = os.getenv("CREATIVE_AZURE_ENDPOINT_RATIOS", "")
        endpoint_ratios = [int(x.strip()) for x in ratios_raw.split(",") if x.strip()] if ratios_raw else None

        api_version = os.getenv("CREATIVE_AZURE_API_VERSION", "2024-12-01-preview")
        rpm = int(os.getenv("CREATIVE_AZURE_RPM", "1500"))
        temperature = float(os.getenv("CREATIVE_AZURE_TEMPERATURE", "0.0"))
        top_p = float(os.getenv("CREATIVE_AZURE_TOP_P", "1.0"))

        return cls(
            endpoints=endpoints,
            models=models,
            requests_per_minute=rpm,
            api_keys=api_keys,
            api_key_env_vars=api_key_env_vars,
            api_version=api_version,
            endpoint_ratios=endpoint_ratios,
            temperature=temperature,
            top_p=top_p,
        )

    def call_api(self, prompt: str, max_num_tokens: int = 1024) -> Any:
        future = asyncio.run_coroutine_threadsafe(
            self._client.generate_single(
                limiter=self._limiter,
                prompt=prompt,
                max_num_tokens=max_num_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            ),
            self._get_loop(),
        )
        result = future.result()
        if result.content is None:
            return []
        return self._safe_parse(result.content)

    def call_api_batch(self, prompts: list[str], max_num_tokens: int = 1024) -> list[Any]:
        future = asyncio.run_coroutine_threadsafe(
            self._client.batch_generate(
                limiter=self._limiter,
                prompts=prompts,
                max_num_tokens=max_num_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            ),
            self._get_loop(),
        )
        results = future.result()
        outputs: list[Any] = []
        for res in results:
            if res.content is None:
                outputs.append("")
            else:
                outputs.append(self._safe_parse(res.content))
        return outputs

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._loop_thread.start()
        return self._loop

    @staticmethod
    def _safe_parse(text: str) -> Any:
        # Match previous creative.py behavior: return parsed JSON-like data when possible.
        if json_repair is not None:
            try:
                return json_repair.loads(text)
            except Exception:
                pass

        try:
            return json.loads(text)
        except Exception:
            return text