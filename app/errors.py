class AppError(Exception):
    """Base error carrying an HTTP status and a stable error code."""

    status = 500
    code = "internal_error"

    def __init__(self, message=None, status=None, code=None, details=None):
        super().__init__(message or self.__class__.__doc__)
        self.message = message or "Unexpected error"
        if status is not None:
            self.status = status
        if code is not None:
            self.code = code
        self.details = details

    def to_dict(self):
        body = {"error": self.code, "message": self.message}
        if self.details:
            body["details"] = self.details
        return body


class AuthError(AppError):
    status = 401
    code = "unauthorized"


class ForbiddenError(AppError):
    status = 403
    code = "forbidden"


class NotFoundError(AppError):
    status = 404
    code = "not_found"


class InsufficientTokens(AppError):
    status = 402
    code = "insufficient_tokens"


class UpstreamError(AppError):
    """The Spring user-management service failed or was unreachable."""

    status = 502
    code = "upstream_unavailable"
