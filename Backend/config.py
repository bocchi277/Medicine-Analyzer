import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # Current generally-available price/performance model. Override via env if
    # you want a different one; the old 'gemini-1.5-pro-latest' is retired.
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    SBERT_MODEL_NAME = os.getenv('SBERT_MODEL_NAME', 'paraphrase-MiniLM-L3-v2')
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "index.faiss")
    ID_MAP_PATH = os.getenv("ID_MAP_PATH", "id_map.pkl")
    DATA_PATH = os.getenv("DATA_PATH", "Drug_Data.csv")

    # The data file is a hard requirement: nothing works without it.
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

    # A missing API key is not fatal. /drug_details and /compare_medicines are
    # fully functional without an LLM; only the recommendation text needs one.
    if not GEMINI_API_KEY:
        logging.warning(
            "GEMINI_API_KEY is not set. Drug lookup and comparison will work, "
            "but /get_alternative will return results without an LLM recommendation."
        )


# Instantiate config
config = Config()
