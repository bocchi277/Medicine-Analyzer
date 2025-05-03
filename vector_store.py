import faiss
import numpy as np
import pickle
import os
import logging
from sentence_transformers import SentenceTransformer
from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_sbert_model():
    """Loads the SentenceTransformer model."""
    logging.info(f"Loading SentenceTransformer model: {config.SBERT_MODEL_NAME}")
    try:
        model = SentenceTransformer(config.SBERT_MODEL_NAME)
        return model
    except Exception as e:
        logging.error(f"Failed to load SBERT model: {e}")
        raise

def build_and_save_index(df, model):
    """Generates embeddings, builds a FAISS index, and saves it along with the ID map."""
    if df.empty or 'combined_embedding_text' not in df.columns:
        logging.error("DataFrame is empty or missing 'combined_embedding_text' column.")
        return None, None

    texts_to_embed = df['combined_embedding_text'].tolist()
    if not texts_to_embed:
        logging.error("No text data found for embedding.")
        return None, None

    logging.info(f"Generating embeddings for {len(texts_to_embed)} texts...")
    try:
        embeddings = model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
    except Exception as e:
        logging.error(f"Error during model encoding: {e}")
        raise

    if embeddings is None or embeddings.shape[0] == 0:
        logging.error("Embedding generation failed or resulted in empty embeddings.")
        return None, None

    dimension = embeddings.shape[1]
    logging.info(f"Embeddings generated with dimension: {dimension}")

    # Build FAISS index (IndexFlatL2 uses Euclidean distance, IndexFlatIP uses Inner Product/Cosine Similarity)
    # For normalized embeddings (like SBERT often produces), IP is equivalent to Cosine Similarity
    index = faiss.IndexFlatIP(dimension)
    # Normalize embeddings for cosine similarity using IndexFlatIP
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    logging.info(f"FAISS index built with {index.ntotal} vectors.")

    # Create mapping from index position to original drug_id
    id_map = {i: drug_id for i, drug_id in enumerate(df.index.to_list())}

    # Save index and map
    try:
        logging.info(f"Saving FAISS index to {config.FAISS_INDEX_PATH}")
        faiss.write_index(index, config.FAISS_INDEX_PATH)
        logging.info(f"Saving ID map to {config.ID_MAP_PATH}")
        with open(config.ID_MAP_PATH, 'wb') as f:
            pickle.dump(id_map, f)
        return index, id_map
    except Exception as e:
        logging.error(f"Error saving index or map: {e}")
        raise

def load_index_and_map():
    """Loads the FAISS index and ID map from disk."""
    if not os.path.exists(config.FAISS_INDEX_PATH) or not os.path.exists(config.ID_MAP_PATH):
        logging.warning("FAISS index or ID map file not found.")
        return None, None
    try:
        logging.info(f"Loading FAISS index from {config.FAISS_INDEX_PATH}")
        index = faiss.read_index(config.FAISS_INDEX_PATH)
        logging.info(f"Loading ID map from {config.ID_MAP_PATH}")
        with open(config.ID_MAP_PATH, 'rb') as f:
            id_map = pickle.load(f)
        logging.info(f"Index loaded with {index.ntotal} vectors. Map loaded with {len(id_map)} entries.")
        return index, id_map
    except Exception as e:
        logging.error(f"Error loading index or map: {e}")
        return None, None

def search_similar(query_embedding, index, id_map, top_n=4):
    """Searches the FAISS index for similar items."""
    if index is None or query_embedding is None:
        logging.error("Cannot search with invalid index or query embedding.")
        return []
    if index.ntotal == 0:
        logging.warning("Attempting to search an empty FAISS index.")
        return []

    try:
        # Normalize the query embedding for cosine similarity search
        query_embedding_norm = np.array([query_embedding], dtype='float32')
        faiss.normalize_L2(query_embedding_norm)

        # Search the index (returns distances and indices)
        # We add 1 to top_n because the query itself might be the most similar
        distances, indices = index.search(query_embedding_norm, top_n + 1)

        results = []
        if len(indices[0]) > 0:
            for i, idx in enumerate(indices[0]):
                if idx != -1: # FAISS can return -1 for invalid indices
                    original_id = id_map.get(idx)
                    if original_id:
                        results.append({
                            'drug_id': original_id,
                            'similarity_score': float(distances[0][i]) # Cosine similarity score
                        })
            # Exclude the query itself if it's found (usually the top result)
            # This assumes the query drug exists in the index
            # A more robust check might involve comparing drug_ids if the query_id is known
            if results and results[0]['similarity_score'] > 0.999: # Check if top result is likely the query itself
                 # Check based on ID if query ID is passed, otherwise assume first is self
                 # For now, let's just remove the top one assuming it's self if score is near 1
                 results = results[1:]


        return results[:top_n] # Return only the requested top_n
    except Exception as e:
        logging.error(f"Error during FAISS search: {e}")
        return []