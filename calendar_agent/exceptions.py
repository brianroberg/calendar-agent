"""Custom exceptions for the Calendar Agent."""


class ProxyError(Exception):
    """Base exception for proxy-related errors."""

    pass


class ProxyAuthError(ProxyError):
    """Raised when authentication with the proxy fails (401)."""

    pass


class ProxyForbiddenError(ProxyError):
    """Raised when an operation is forbidden or requires confirmation (403)."""

    pass


class LLMError(Exception):
    """Raised when LLM operations fail."""

    pass
