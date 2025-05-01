import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    SBERT_MODEL_NAME = os.getenv("SBERT_MODEL_NAME", 'all-MiniLM-L6-v2')
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "index.faiss")
    ID_MAP_PATH = os.getenv("ID_MAP_PATH", "id_map.pkl")
    DATA_PATH = os.getenv("DATA_PATH", "Drug_Data.csv")

    # Basic validation
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    if not os.path.exists(DATA_PATH):
         raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

# Instantiate config
config = Config()