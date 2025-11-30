from flask import Flask, jsonify
from flask_cors import CORS
import logging
import os

# Import components from other modules
from config import config
from data_processing import load_and_preprocess_data
from vector_store import get_sbert_model, build_and_save_index, load_index_and_map
from api import api_bp, init_api

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__)
    CORS(app)

    try:
        logging.info("--- Initializing Application ---")
        drug_data = load_and_preprocess_data()
        sbert_model = get_sbert_model()

        faiss_index, faiss_id_map = load_index_and_map()
        if faiss_index is None or faiss_id_map is None:
            logging.warning("Building new FAISS index...")
            if not drug_data.empty and sbert_model:
                faiss_index, faiss_id_map = build_and_save_index(drug_data, sbert_model)
                if faiss_index is None:
                    raise RuntimeError("Failed to build FAISS index.")
            else:
                raise RuntimeError("Missing data or model for index building.")
        else:
            logging.info("Loaded existing FAISS index.")

        if faiss_index.ntotal != len(faiss_id_map):
            logging.warning("Index mismatch. Rebuilding...")
            faiss_index, faiss_id_map = build_and_save_index(drug_data, sbert_model)

        init_api(sbert_model, faiss_index, faiss_id_map, drug_data)
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

    return app

# Create the app instance for Gunicorn
app = create_app()

if __name__ == '__main__':
    # Render uses PORT environment variable
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 8000))
    app.run(host=HOST, port=PORT, debug=False)