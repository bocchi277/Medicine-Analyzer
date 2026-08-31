from flask import Flask, jsonify
from flask_cors import CORS
import logging
import os

# Import components from other modules
from config import config
from data_processing import load_and_preprocess_data
from vector_store import load_index_and_map
from api import api_bp, init_api

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)

    # Restrict cross-origin access to the known frontends rather than '*'.
    #
    # The API is public and read-only, so a wildcard was not a data-leak risk,
    # but it did let any site drive our Gemini quota using a visitor's IP as the
    # rate-limit key. Naming the origins keeps that cost tied to our own pages.
    #
    # ALLOWED_ORIGINS overrides this list (comma-separated) so a new domain or
    # Netlify preview URL needs an env var change, not a redeploy of code.
    default_origins = [
        "https://medsphere707.netlify.app",   # production frontend
        "http://localhost:8899",              # local static server
        "http://127.0.0.1:8899",
        "http://localhost:5500",              # VS Code Live Server
        "http://127.0.0.1:5500",
    ]
    configured = os.environ.get("ALLOWED_ORIGINS", "").strip()
    origins = [o.strip() for o in configured.split(",") if o.strip()] or default_origins

    # Netlify deploy previews live on <hash>--medsphere707.netlify.app.
    origin_patterns = origins + [r"https://.*--medsphere707\.netlify\.app"]

    CORS(app, resources={r"/api/*": {"origins": origin_patterns},
                         r"/healthz": {"origins": "*"}})
    logging.info(f"CORS allowed origins: {origins} (+ Netlify deploy previews)")

    try:
        logging.info("--- Initializing Application ---")

        # The index is a build artifact, produced offline by build_index.py.
        # Building it here would require sentence-transformers/PyTorch in the
        # web process, which alone exceeds a 512 MB memory budget.
        faiss_index, faiss_id_map, faiss_positions = load_index_and_map()
        if faiss_index is None:
            raise RuntimeError(
                "No usable FAISS index found. Run 'python build_index.py' locally "
                "(see requirements-build.txt) and deploy the generated "
                f"'{config.FAISS_INDEX_PATH}' and '{config.ID_MAP_PATH}'."
            )
        logging.info("Loaded existing FAISS index.")

        drug_data = load_and_preprocess_data()

        init_api(faiss_index, faiss_id_map, faiss_positions, drug_data)
        # Release this scope's reference so the only one left is inside the api
        # module; otherwise the frame stays alive in the closure below.
        del drug_data
        app.register_blueprint(api_bp)
        logging.info("API Blueprint registered.")

    except Exception as e:
        logging.exception("Initialization failed.")
        raise SystemExit(f"Error: {e}") from e

    @app.route('/')
    def index():
        return jsonify({
            "message": "Alternative Medicine Recommendation API",
            "status": "OK",
            "vector_index_size": faiss_index.ntotal if faiss_index else 0
        })

    @app.route('/healthz')
    def healthz():
        """Cheap liveness probe that avoids waking any heavy code paths."""
        return jsonify({"status": "ok"}), 200

    return app


# Create the app instance for Gunicorn
app = create_app()

if __name__ == '__main__':
    # Render uses PORT environment variable
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 8000))
    app.run(host=HOST, port=PORT, debug=False)
