import logging

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from config import Config

from .auth import TokenVerifier
from .auth_client import AuthServiceClient
from .dispatch import create_dispatcher
from .errors import AppError
from .flux import FluxClient
from .image_client import ImageServiceClient
from .job_store import create_job_store
from .logctx import install as install_log_context
from .storage import create_storage


def create_app(config=None):
    app = Flask(__name__)
    config = config or Config()
    app.config["APP_CONFIG"] = config
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES
    app.extensions["verifier"] = TokenVerifier(config)
    app.extensions["auth_client"] = AuthServiceClient(config)
    app.extensions["flux"] = FluxClient(config)
    app.extensions["images"] = ImageServiceClient(config)
    # Generation jobs: where they are recorded, where their images go, and what
    # runs them. All three follow config.TASKS_MODE (local | inline | cloud).
    app.extensions["jobs"] = create_job_store(config)
    app.extensions["storage"] = create_storage(config)
    app.extensions["dispatcher"] = create_dispatcher(app, config)

    # [job] is the first 8 characters of the generation job id, "-" outside a
    # job (app/logctx.py). grep for it to see one generation end to end.
    install_log_context()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(job)s] %(name)s %(message)s",
    )

    from .generate_routes import bp as generate_bp
    from .job_routes import bp as jobs_bp
    from .routes import bp

    app.register_blueprint(bp)
    app.register_blueprint(generate_bp)
    app.register_blueprint(jobs_bp)
    app.logger.info("generation jobs: TASKS_MODE=%s, JOB_STORE=%s, STORAGE_BACKEND=%s",
                    config.TASKS_MODE, config.JOB_STORE, config.STORAGE_BACKEND)
    _register_error_handlers(app)
    return app


def _register_error_handlers(app):
    @app.errorhandler(AppError)
    def handle_app_error(exc):
        # Every refused request is logged with its reason. Without this a 4xx left
        # only werkzeug's access line ("POST /api/generate 403") and no way to
        # tell a bad field from a refused balance check.
        details = f" details={exc.details}" if exc.details else ""
        level = logging.ERROR if exc.status >= 500 else logging.WARNING
        app.logger.log(level, "%s %s -> %d %s: %s%s", request.method, request.path,
                       exc.status, exc.code, exc.message, details)
        response = jsonify(exc.to_dict())
        # RFC 6750: a 401 from a bearer-token resource says so in the header.
        if exc.status == 401:
            response.headers["WWW-Authenticate"] = (
                f'Bearer realm="{app.config["APP_CONFIG"].JWT_AUDIENCE}", '
                f'error="invalid_token"'
            )
        return response, exc.status

    @app.errorhandler(HTTPException)
    def handle_http_error(exc):
        return jsonify(error=exc.name.lower().replace(" ", "_"), message=exc.description), exc.code

    @app.errorhandler(Exception)
    def handle_unexpected(exc):
        app.logger.exception("Unhandled error")
        return jsonify(error="internal_error", message="Unexpected server error"), 500
