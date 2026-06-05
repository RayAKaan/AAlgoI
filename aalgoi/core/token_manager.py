import os
import logging

logger = logging.getLogger(__name__)

_ENV_VAR = "GITHUB_TOKEN"
_KEYRING_SERVICE = "aalgoi"
_KEYRING_USER = "github_token"
_cache = {"token": None}

try:
    import keyring as _keyring
    _KEYRING_AVAILABLE = True
except ImportError:
    _KEYRING_AVAILABLE = False


class TokenManager:
    """
    Reads, caches, and validates the GitHub token.

    Priority:
        1. Operating system keyring (keyring package)
        2. GITHUB_TOKEN environment variable

    Usage:
        token = TokenManager.get_token()        # None if unset
        token = TokenManager.require_token()     # raises ValueError if unset
        TokenManager.clear_cache()               # force re-read on next call
        TokenManager.store_token("ghp_...")      # persist to keyring
        TokenManager.delete_token()              # remove from keyring
    """

    @staticmethod
    def get_token() -> str | None:
        if _cache["token"] is None:
            _cache["token"] = TokenManager._read_token()
        return _cache["token"]

    @staticmethod
    def require_token() -> str:
        token = TokenManager.get_token()
        if not token:
            raise ValueError(
                f"GitHub token not found. "
                f"Set the {_ENV_VAR} environment variable, "
                f"or use TokenManager.store_token()."
            )
        return token

    @staticmethod
    def store_token(token: str) -> None:
        if _KEYRING_AVAILABLE:
            _keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, token)
            logger.info("Token stored in system keyring")
        else:
            logger.warning("keyring not available; token only cached in memory")
        _cache["token"] = token

    @staticmethod
    def delete_token() -> None:
        if _KEYRING_AVAILABLE:
            try:
                _keyring.delete_password(_KEYRING_SERVICE, _KEYRING_USER)
                logger.info("Token removed from system keyring")
            except _keyring.errors.PasswordDeleteError:
                pass
        _cache["token"] = None

    @staticmethod
    def clear_cache() -> None:
        _cache["token"] = None

    @staticmethod
    def _read_token() -> str | None:
        if _KEYRING_AVAILABLE:
            try:
                token = _keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER)
                if token:
                    logger.info("Token found in system keyring")
                    return token
            except Exception:
                pass
        token = os.getenv(_ENV_VAR) or None
        if token:
            logger.info("Token found in environment variable")
        return token
