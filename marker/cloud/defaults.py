"""Shared defaults for quality-oriented Ollama Cloud document recognition.

The HTTP API uses ``kimi-k3``; ``kimi-k3:cloud`` is Ollama's CLI alias.
These budgets bound individual objects rather than combining unrelated tables.
"""

DEFAULT_CLOUD_MODEL = "kimi-k3"
DEFAULT_CLOUD_BASE_URL = "https://ollama.com/v1"
DEFAULT_CONTEXT_TOKENS = 245760
DEFAULT_MAX_OUTPUT_TOKENS = 16384
DEFAULT_MAX_IMAGES = 16
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_CONCURRENCY = 2
