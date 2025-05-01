from flask import Flask, jsonify
from flask_cors import CORS
import logging
import os

# Import components from other modules
from config import config
from data_processing import load_and_preprocess_data
from vector_store import get_sbert_model, build_and_save_index, load_index_and_map
from api import api_bp, init_api # Import 

# --- Basic Logging Setup ---
# In production, configure this more robustly (e.g., rotating file handlers, different levels)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)
    CORS(app) # Enable CORS for all routes

    # --- Load Data and Models ---
    # Wrap in try-except to handle potential errors during startup
    try:
        logging.info("--- Initializing Application ---")
        # 1. Load and preprocess data
        drug_data = load_and_preprocess_data()
        # After loading drug_data
        # 2. Load SBERT model
        sbert_model = get_sbert_model()

        # 3. Load or Build FAISS Index
        faiss_index, faiss_id_map = load_index_and_map()
        if faiss_index is None or faiss_id_map is None:
            logging.warning("FAISS index not found or failed to load. Building new index...")
            if not drug_data.empty and sbert_model:
                faiss_index, faiss_id_map = build_and_save_index(drug_data, sbert_model)
                if faiss_index is None:
                     raise RuntimeError("Failed to build FAISS index.")
            else:
                raise RuntimeError("Cannot build index without data and model.")
        else:
            logging.info("Loaded existing FAISS index and ID map.")

        # Check consistency
        if faiss_index.ntotal != len(faiss_id_map):
             logging.warning(f"Index size ({faiss_index.ntotal}) mismatch with map size ({len(faiss_id_map)}). Rebuilding index.")
             # Force rebuild if inconsistent
             faiss_index, faiss_id_map = build_and_save_index(drug_data, sbert_model)
             if faiss_index is None:
                 raise RuntimeError("Failed to rebuild FAISS index after inconsistency.")

        # --- Initialize API with loaded resources ---
        # This makes the loaded objects available to the routes in api.py
        init_api(sbert_model, faiss_index, faiss_id_map, drug_data)

        # --- Register Blueprints ---
        app.register_blueprint(api_bp)
        logging.info("API Blueprint registered.")

        logging.info("--- Application Initialization Complete ---")

    except FileNotFoundError as e:
        logging.error(f"Fatal Error during init: {e}. Ensure '{config.DATA_PATH}' exists.")
        # Exit or raise to prevent running without data
        raise SystemExit(f"Initialization failed: {e}") from e
    except ValueError as e:
         logging.error(f"Fatal Error during init: {e}.")
         raise SystemExit(f"Initialization failed: {e}") from e
    except Exception as e:
        logging.exception("An unexpected error occurred during application initialization.")
        # Depending on severity, might want to exit
        raise SystemExit(f"Unexpected initialization error: {e}") from e


    # --- Basic Root Route ---
    @app.route('/')
    def index():
        # Simple health check or info endpoint
        return jsonify({
            "message": "Alternative Medicine Recommendation API",
            "status": "OK",
            "vector_index_size": faiss_index.ntotal if faiss_index else 0
            })

    return app

# --- Main Execution ---
if __name__ == '__main__':
    # Get host and port from environment variables or use defaults
    # Defaults are suitable for development, Gunicorn/WSGI server will handle this in production
    HOST = os.environ.get('FLASK_RUN_HOST', '127.0.0.1')
    try:
        PORT = int(os.environ.get('FLASK_RUN_PORT', '8000'))
    except ValueError:
        PORT = 8000

    # Create the app instance
    # Errors during create_app() will stop execution here
    app = create_app()

    # DO NOT use app.run(debug=True) in production!
    # Use a WSGI server like Gunicorn:
    # gunicorn --bind 0.0.0.0:8000 app:app
    logging.info(f"Starting Flask development server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False) # debug=False is safer even for local testing