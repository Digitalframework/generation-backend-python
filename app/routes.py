from flask import Blueprint, current_app, g, jsonify, request

from .auth import require_auth, require_role
from .errors import AppError

bp = Blueprint("api", __name__, url_prefix="/api")


def _client():
    return current_app.extensions["auth_client"]


@bp.route("/health", methods=["GET"])
def health():
    reachable = _client().health()
    return jsonify(status="ok", authService="up" if reachable else "down")


@bp.route("/me", methods=["GET"])
@require_auth
def me():
    """Whoever the token says the caller is. No network call — the token carries it."""
    return jsonify(id=g.user["id"], email=g.user["email"], roles=g.user["roles"])


@bp.route("/tokens/balance", methods=["GET"])
@require_auth
def balance():
    return jsonify(userId=g.user["id"], balance=_client().get_balance(g.token))


@bp.route("/tokens/check", methods=["GET"])
@require_auth
def check():
    """Non-destructive: can the caller afford `amount` tokens?"""
    amount = _positive_int_arg("amount", default=1)
    current = _client().get_balance(g.token)
    return jsonify(
        userId=g.user["id"],
        balance=current,
        required=amount,
        sufficient=current >= amount,
    )


@bp.route("/tokens/spend", methods=["POST"])
@require_auth
def spend():
    payload = request.get_json(silent=True) or {}
    amount = payload.get("amount", 1)
    if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
        raise AppError("`amount` must be a positive integer", status=400, code="bad_request")

    remaining = _client().spend(g.token, amount, reason=payload.get("reason"))
    return jsonify(userId=g.user["id"], spent=amount, balance=remaining)


@bp.route("/admin/health", methods=["GET"])
@require_auth
@require_role("ADMIN")
def admin_health():
    """Example of a role-gated route. Roles come from the token's `roles` claim."""
    return jsonify(status="ok", caller=g.user["email"])


def _positive_int_arg(name, default):
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise AppError(f"`{name}` must be an integer", status=400, code="bad_request") from None
    if value <= 0:
        raise AppError(f"`{name}` must be positive", status=400, code="bad_request")
    return value
