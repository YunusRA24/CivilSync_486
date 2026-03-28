"""
Flask application factory for CivilSync.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask


def create_app():
    app = Flask(__name__)
    app.secret_key = "civilsync-486-dev-key"

    from webapp.routes import bp
    app.register_blueprint(bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5050)
