"""
Runtime vector store: loads a prebuilt FAISS index and searches it.

Deliberately free of sentence-transformers / PyTorch. Every drug the API
looks up already exists in the index, so its embedding can be read back
with index.reconstruct(position) instead of being recomputed. That keeps
the server's resident memory in the ~180 MB range instead of ~600 MB.

Index building lives in build_index.py and runs offline.
"""

import logging
import os
import pickle

import faiss
import numpy as np

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_index_and_map():
    """
    Load the FAISS index plus both directions of the position/drug_id map.

    Returns:
        (index, id_map, position_by_drug_id) or (None, None, None) on failure.
        id_map maps index position -> drug_id; position_by_drug_id is its
        inverse, used to fetch a drug's stored vector.
    """
    if not os.path.exists(config.FAISS_INDEX_PATH) or not os.path.exists(config.ID_MAP_PATH):
        logging.error(
            "FAISS index or ID map file not found. Run 'python build_index.py' "
            "locally and commit the generated index.faiss and id_map.pkl."
        )
        return None, None, None

    try:
        logging.info(f"Loading FAISS index from {config.FAISS_INDEX_PATH}")
        index = faiss.read_index(config.FAISS_INDEX_PATH)
        logging.info(f"Loading ID map from {config.ID_MAP_PATH}")
        with open(config.ID_MAP_PATH, "rb") as f:
            id_map = pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading index or map: {e}")
        return None, None, None

    if index.ntotal != len(id_map):
        logging.error(
            f"Index/map size mismatch: index has {index.ntotal} vectors but the map "
            f"has {len(id_map)} entries. Rebuild with 'python build_index.py'."
        )
        return None, None, None

    position_by_drug_id = {drug_id: pos for pos, drug_id in id_map.items()}
    if len(position_by_drug_id) != len(id_map):
        logging.warning(
            "Duplicate drug_ids in the ID map; some drugs will resolve to a single "
            "index position. Rebuild the index from de-duplicated data."
        )

    logging.info(
        f"Index loaded: {index.ntotal} vectors, dim {index.d}, {len(id_map)} mapped ids."
    )
    return index, id_map, position_by_drug_id


def get_stored_vector(index, position):
    """
    Read a drug's embedding straight out of the index.

    Replaces re-running the SBERT model over text that was already embedded
    at build time. Returns a unit-length float32 vector of shape (1, d),
    ready for both search and dot-product comparison.
    """
    if index is None or position is None:
        return None
    if not 0 <= position < index.ntotal:
        logging.error(f"Vector position {position} out of range (ntotal={index.ntotal}).")
        return None

    vector = np.array([index.reconstruct(int(position))], dtype="float32")
    # Vectors are normalised at build time; renormalising is a ~microsecond
    # no-op that keeps this correct even against an index built differently.
    faiss.normalize_L2(vector)
    return vector


def cosine_similarity_of(vector_a, vector_b):
    """Cosine similarity between two unit-length (1, d) vectors."""
    if vector_a is None or vector_b is None:
        return None
    return float(np.dot(vector_a[0], vector_b[0]))


def search_similar(query_vector, index, id_map, top_n=4, exclude_position=None):
    """
    Find the nearest drugs to a query vector.

    Args:
        query_vector: unit-length array of shape (1, d), from get_stored_vector.
        exclude_position: index position of the query drug itself, so it can be
            dropped by identity rather than by guessing from its score.

    Returns:
        List of {'drug_id', 'similarity_score'}, highest similarity first.
    """
    if index is None or query_vector is None:
        logging.error("Cannot search with an invalid index or query vector.")
        return []
    if index.ntotal == 0:
        logging.warning("Attempting to search an empty FAISS index.")
        return []

    try:
        # One extra result to absorb the query drug matching itself.
        distances, indices = index.search(query_vector, min(top_n + 1, index.ntotal))

        results = []
        for score, position in zip(distances[0], indices[0]):
            if position == -1 or position == exclude_position:
                continue
            drug_id = id_map.get(int(position))
            if drug_id is not None:
                results.append({"drug_id": drug_id, "similarity_score": float(score)})

        return results[:top_n]
    except Exception as e:
        logging.error(f"Error during FAISS search: {e}")
        return []
