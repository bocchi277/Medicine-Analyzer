"""
Offline FAISS index builder.

Run this LOCALLY (not on the server) whenever Drug_Data.csv changes:

    pip install -r requirements-build.txt
    python build_index.py

It writes index.faiss and id_map.pkl, which the web server then loads
read-only. This is the only part of the project that needs
sentence-transformers / PyTorch, which is why it lives outside the
runtime dependency set: the server would need ~400 MB of RAM just to
import torch, and it never actually needs to embed anything at request
time (every vector it looks up is already stored in the index).
"""

import logging
import pickle

import faiss
import numpy as np

from config import config
from data_processing import load_and_preprocess_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_sbert_model():
    """Load the SentenceTransformer model used to embed drug text."""
    from sentence_transformers import SentenceTransformer

    logging.info(f"Loading SentenceTransformer model: {config.SBERT_MODEL_NAME}")
    return SentenceTransformer(config.SBERT_MODEL_NAME)


def build_and_save_index(df, model):
    """
    Embed every row's combined text, build a cosine-similarity FAISS index,
    and persist it alongside a position -> drug_id map.

    Vectors are L2-normalised before insertion, so IndexFlatIP's inner
    product is exactly cosine similarity. The runtime depends on this:
    it reads vectors straight back out with index.reconstruct() instead of
    re-embedding, and assumes they are already unit length.
    """
    if df.empty or "combined_embedding_text" not in df.columns:
        raise ValueError("DataFrame is empty or missing 'combined_embedding_text'.")

    texts = df["combined_embedding_text"].tolist()
    if not texts:
        raise ValueError("No text data found for embedding.")

    logging.info(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    if embeddings is None or embeddings.shape[0] == 0:
        raise ValueError("Embedding generation produced no vectors.")

    embeddings = np.ascontiguousarray(embeddings, dtype="float32")
    dimension = embeddings.shape[1]
    logging.info(f"Embeddings generated with dimension {dimension}.")

    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    logging.info(f"FAISS index built with {index.ntotal} vectors.")

    # Position in the index -> original drug_id. Must stay in sync with the
    # row order used above, which is why both are written together.
    id_map = {i: drug_id for i, drug_id in enumerate(df.index.to_list())}

    logging.info(f"Saving FAISS index to {config.FAISS_INDEX_PATH}")
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    logging.info(f"Saving ID map to {config.ID_MAP_PATH}")
    with open(config.ID_MAP_PATH, "wb") as f:
        pickle.dump(id_map, f)

    return index, id_map


def main():
    drug_data = load_and_preprocess_data(for_index=True)
    model = get_sbert_model()
    index, id_map = build_and_save_index(drug_data, model)
    logging.info(
        f"Done. index.ntotal={index.ntotal}, id_map entries={len(id_map)}, dim={index.d}"
    )


if __name__ == "__main__":
    main()
