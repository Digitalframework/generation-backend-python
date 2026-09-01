import logging

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from config import Config

from .auth import TokenVerifier
from .auth_client import AuthServiceClient
from .errors import AppError
from .flux import FluxClient
from .image_client import ImageServiceClient


def create_app(config=None):
    app = Flask(__name__)
    config = config or Config()
    app.config["APP_CONFIG"] = config
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES
    app.extensions["verifier"] = TokenVerifier(config)
    app.extensions["auth_client"] = AuthServiceClient(config)
    app.extensions["flux"] = FluxClient(config)
    app.extensions["images"] = ImageServiceClient(config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from .generate_routes import bp as generate_bp
    from .routes import bp

    app.register_blueprint(bp)
    app.register_blueprint(generate_bp)
    _register_error_handlers(app)
    return app


def _register_error_handlers(app):
    @app.errorhandler(AppError)
    def handle_app_error(exc):
        if exc.status >= 500:
            app.logger.error("%s: %s", exc.code, exc.message)
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
