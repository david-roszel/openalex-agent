"""Environment configuration; secrets are excluded from representations."""

from dataclasses import dataclass, field
import math
import os
from dotenv import load_dotenv


@dataclass
class Settings:
    database: str = "data/openalex.sqlite"
    api_key: str = field(default="", repr=False)
    email: str = field(default="", repr=False)
    timeout: float = 30.0
    min_interval: float = 1.0
    max_requests: int = 500
    max_retries: int = 3
    max_cost_usd: float = 0.10
    llm_base_url: str = ""
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = ""
    llm_input_price: float = 0.0
    llm_output_price: float = 0.0
    llm_timeout: float = 30.0
    llm_max_tokens: int = 500
    llm_temperature: float | None = 0.0
    llm_reasoning_effort: str = ""

    def validate_retrieval(self) -> None:
        """Reject unsafe retrieval limits before any network activity."""
        for name, minimum in (("timeout", 0), ("min_interval", 0), ("max_cost_usd", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.timeout == 0:
            raise ValueError("timeout must be greater than zero")
        for name in ("max_requests", "max_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        def number(name: str, default: str, cast):
            try:
                return cast(os.getenv(name, default))
            except (ValueError, OverflowError):
                raise ValueError(f"Invalid numeric configuration: {name}") from None

        settings = cls(
            database=os.getenv("OPENALEX_DB", "data/openalex.sqlite"),
            api_key=os.getenv("OPENALEX_API_KEY", ""),
            email=os.getenv("OPENALEX_EMAIL", ""),
            timeout=number("OPENALEX_REQUEST_TIMEOUT", "30", float),
            min_interval=number("OPENALEX_MIN_INTERVAL", "1", float),
            max_requests=number("OPENALEX_MAX_REQUESTS", "500", int),
            max_retries=number("OPENALEX_MAX_RETRIES", "3", int),
            max_cost_usd=number("OPENALEX_MAX_COST_USD", "0.10", float),
            llm_base_url=os.getenv("LLM_BASE_URL", ""),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", ""),
            llm_input_price=number("LLM_INPUT_USD_PER_MILLION", "0", float),
            llm_output_price=number("LLM_OUTPUT_USD_PER_MILLION", "0", float),
            llm_timeout=number("LLM_TIMEOUT", "30", float),
            llm_max_tokens=number("LLM_MAX_OUTPUT_TOKENS", "500", int),
            llm_temperature=None if os.getenv("LLM_TEMPERATURE", "0").strip().lower() == "omit"
                else number("LLM_TEMPERATURE", "0", float),
            llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "").strip(),
        )
        settings.validate_retrieval()
        return settings
