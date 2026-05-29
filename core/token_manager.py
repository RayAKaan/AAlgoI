import os
import logging

logger = logging.getLogger(__name__)

_ENV_VAR = "GITHUB_TOKEN"
_cache = {"token": None}


class TokenManager:
    """
    Reads, caches, and validates the GITHUB_TOKEN environment variable.

    Usage:
        token = TokenManager.get_token()        # None if unset
        token = TokenManager.require_token()     # raises ValueError if unset
        TokenManager.clear_cache()               # force re-read on next call
    """

    @staticmethod
    def get_token() -> str | None:
        if _cache["token"] is None:
            _cache["token"] = os.getenv(_ENV_VAR) or None
            if _cache["token"]:
                logger.info("GITHUB_TOKEN found in environment")
        return _cache["token"]

    @staticmethod
    def require_token() -> str:
        token = TokenManager.get_token()
        if not token:
            raise ValueError(
                f"GitHub token not found. "
                f"Set the {_ENV_VAR} environment variable."
            )
        return token

    @staticmethod
    def clear_cache():
        _cache["token"] = None
