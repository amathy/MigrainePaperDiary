"""WSGI entry point: ``gunicorn wsgi:app``."""

from webapp.app import app  # noqa: F401

if __name__ == "__main__":
    import os

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
