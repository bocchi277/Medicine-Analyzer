from flask import Blueprint, request, jsonify
import logging
import numpy as np # Import numpy
from llm_service import get_best_alternative_from_llm # Use absolute import
from vector_store import search_similar # Use absolute import
import pandas as pd # Import pandas for DataFrame operations
from sklearn.metrics.pairwise import cosine_similarity # Import for similarity comparison

# Ensure logging is configured to show INFO level or higher
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# These would be passed during app creation/initialization
# Global or passed via app context is okay for this scale, dependency injection frameworks are better for large apps
sbert_model = None
faiss_index = None
faiss_id_map = None
drug_data = None # The full processed DataFrame

def init_api(model, index, id_map, data_df):
    """Initialize globals needed by the API routes."""
    global sbert_model, faiss_index, faiss_id_map, drug_data
    sbert_model = model
    faiss_index = index
    faiss_id_map = id_map
    logging.info("API resources initialized.")
    # IMPORTANT: Ensure drug_data is not None and has expected columns after init
    if data_df is None or data_df.empty:
        logging.error("Data DataFrame is None or empty after initialization!")
        drug_data = pd.DataFrame() # Ensure it's a DataFrame even if empty
    else:
        drug_data = data_df
        logging.info(f"API initialized with data shape: {drug_data.shape}")
        logging.info(f"API initialized with data columns: {drug_data.columns.tolist()}")


def _find_drug_by_name(medicine_name: str):
    """
    Helper function to find a drug in the DataFrame by name, generic name,
    brand name, or synonym (case-insensitive).

    Args:
        medicine_name (str): The name to search for.

    Returns:
        pd.Series or None: The DataFrame row for the found drug, or None if not found.
    """
    if drug_data is None or drug_data.empty:
        logging.error("_find_drug_by_name called before drug_data is initialized.")
        return None

    # Case-insensitive search across name, generic_name, brand_name, and synonyms
    search_cols = ['name', 'generic_name']
    if 'brand_name' in drug_data.columns:
        search_cols.append('brand_name')
    if 'synonyms' in drug_data.columns:
        search_cols.append('synonyms')

    search_condition = pd.Series([False] * len(drug_data), index=drug_data.index)
    input_name_lower = medicine_name.lower()

    for col in search_cols:
        if col in drug_data.columns and pd.api.types.is_string_dtype(drug_data[col]):
            if col in ['name', 'generic_name', 'brand_name']:
                current_col_condition = (drug_data[col].str.lower() == input_name_lower)
                search_condition = search_condition | current_col_condition
            elif col == 'synonyms':
                current_col_condition = (drug_data[col].str.lower().str.contains(input_name_lower, na=False))
                search_condition = search_condition | current_col_condition

    target_drug_series = drug_data[search_condition]

    if target_drug_series.empty:
        return None # Drug not found

    # Return the first match if multiple are found
    return target_drug_series.iloc[0]


# --- Existing Route: Get Alternative ---
@api_bp.route('/get_alternative', methods=['GET'])
def get_alternative_route():
    """
    API endpoint to find similar drugs and the best alternative.
    Allows searching by medicine name, generic name, brand name, or synonym.
    """
    medicine_name = request.args.get('name')
    if not medicine_name:
        logging.warning("Missing 'name' parameter in /get_alternative request.")
        return jsonify({'error': 'Please provide a medicine name parameter (e.g., /api/get_alternative?name=Aspirin)'}), 400

    # Check if resources are initialized
    required_cols = ['drug_id', 'name', 'generic_name', 'combined_embedding_text']
    if drug_data is None or drug_data.empty or not all(col in drug_data.columns for col in required_cols) or sbert_model is None or faiss_index is None or faiss_id_map is None:
         logging.error("/get_alternative: API resources not initialized properly or missing critical data columns.")
         logging.error(f"Data initialized: {drug_data is not None and not drug_data.empty}. Missing cols: {[col for col in required_cols if drug_data is not None and col not in drug_data.columns] if drug_data is not None else 'Data is None'}")
         return jsonify({'error': 'Server not ready, resources not loaded or data incomplete'}), 503 # Service Unavailable


    logging.info(f"Received /get_alternative request for: {medicine_name}")

    # --- Step 1: Find the target drug using the helper function ---
    try:
        target_drug_info = _find_drug_by_name(medicine_name)

        if target_drug_info is None:
            logging.warning(f"/get_alternative: Medicine not found in dataset: {medicine_name}")
            return jsonify({'error': f"Medicine '{medicine_name}' not found. Please check the spelling or try a different name."}), 404

        target_drug_id = target_drug_info['drug_id']
        # Ensure 'combined_embedding_text' exists and is used for embedding
        if 'combined_embedding_text' not in target_drug_info or pd.isna(target_drug_info['combined_embedding_text']) or target_drug_info['combined_embedding_text'] == "":
             logging.error(f"/get_alternative: 'combined_embedding_text' is missing or empty for target drug ID {target_drug_id}. Check data processing.")
             return jsonify({'error': 'Internal server error: Embedding text missing for target drug.'}), 500

        target_text = target_drug_info['combined_embedding_text']
        logging.debug(f"/get_alternative: Target embedding text: {target_text[:100]}...") # Log first 100 chars

        # Generate embedding for the target drug
        if sbert_model is None:
             logging.error("/get_alternative: SBERT model is not initialized.")
             return jsonify({'error': 'Internal server error: Embedding model not loaded.'}), 500

        logging.info("/get_alternative: Generating embedding for target drug.")
        query_embedding = sbert_model.encode(target_text, convert_to_numpy=True)
        if query_embedding is None or not isinstance(query_embedding, np.ndarray) or query_embedding.size == 0:
             raise ValueError("Failed to generate a valid numpy embedding for the query drug.")
        logging.info(f"/get_alternative: Embedding generated with shape: {query_embedding.shape}")


    except Exception as e:
        logging.error(f"/get_alternative: Error finding target drug or generating embedding for '{medicine_name}': {e}", exc_info=True) # Log traceback
        return jsonify({'error': f"An internal error occurred while processing the request for '{medicine_name}'. Details logged on server."}), 500


    # --- Step 2: Find similar medicines using FAISS ---
    logging.info(f"/get_alternative: Searching FAISS index for drugs similar to {target_drug_info['name']} (ID: {target_drug_id})")
    if faiss_index is None or faiss_id_map is None:
         logging.error("/get_alternative: FAISS index or ID map is not initialized.")
         return jsonify({'error': 'Internal server error: Search index not loaded.'}), 503 # Use 503 as it's a server resource issue

    try:
        # search_similar function should handle the FAISS search and mapping
        # Use a reasonable top_n here
        logging.info("Calling search_similar with top_n=10") # Reverted top_n to a more typical value
        similar_indices_scores = search_similar(query_embedding, faiss_index, faiss_id_map, top_n=10)

        # --- DEBUG LOGGING ---
        logging.info(f"/get_alternative: search_similar returned {len(similar_indices_scores)} results.")
        if similar_indices_scores:
             logging.info(f"/get_alternative: Top 5 results from search_similar: {similar_indices_scores[:5]}")
        else:
             logging.info("/get_alternative: search_similar returned an empty list.")
        # --- END DEBUG LOGGING ---


    except Exception as e:
         logging.error(f"/get_alternative: Error during FAISS search: {e}", exc_info=True) # Log traceback
         return jsonify({'error': 'Internal server error during similarity search.'}), 500


    if not similar_indices_scores:
        logging.warning(f"/get_alternative: No similar drugs found in the index for {target_drug_info['name']} within top_n.")
        # Prepare target drug dict, excluding embedding text
        target_drug_dict = target_drug_info.to_dict()
        if 'combined_embedding_text' in target_drug_dict:
            del target_drug_dict['combined_embedding_text']
            logging.debug("/get_alternative: Removed 'combined_embedding_text' from target_drug in response.")

        return jsonify({
            'target_drug': target_drug_dict,
            'similar_medicines': [], # Explicitly empty list
            'best_alternative_recommendation': 'No similar drugs found in the index based on the search criteria.'
        }), 200


    # --- Step 3: Retrieve details for similar medicines ---
    # Extract drug_ids from the search results
    similar_drug_ids_from_search = [res['drug_id'] for res in similar_indices_scores]

    # Filter out the target drug itself from the list of similar drugs
    # This assumes search_similar returns the target drug as the top result with score near 1
    # We keep the explicit ID check for robustness
    similar_drug_ids = [id for id in similar_drug_ids_from_search if id != target_drug_id]

    # --- DEBUG LOGGING ---
    logging.info(f"/get_alternative: After filtering target drug (ID: {target_drug_id}), {len(similar_drug_ids)} similar drug IDs remain.")
    if similar_drug_ids_from_search:
        logging.info(f"/get_alternative: Original IDs from search_similar: {similar_drug_ids_from_search[:5]}")
    if similar_drug_ids:
        logging.info(f"/get_alternative: Filtered similar drug IDs: {similar_drug_ids[:5]}")
    # --- END DEBUG LOGGING ---


    if not similar_drug_ids:
         logging.warning(f"/get_alternative: Only the target drug itself was found as similar for {target_drug_info['name']} after filtering.")
         # Prepare target drug dict, excluding embedding text
         target_drug_dict = target_drug_info.to_dict()
         if 'combined_embedding_text' in target_drug_dict:
             del target_drug_dict['combined_embedding_text']
             logging.debug("/get_alternative: Removed 'combined_embedding_text' from target_drug in response.")

         return jsonify({
             'target_drug': target_drug_dict,
             'similar_medicines': [], # Explicitly empty list
             'best_alternative_recommendation': 'The search found the target medicine but no other distinct similar drugs in the index within the top results.'
         }), 200


    try:
        # Retrieve full data for the similar drugs using their IDs from the main drug_data DataFrame
        # Use .loc for index-based lookup
        # Ensure the IDs in similar_drug_ids actually exist in the DataFrame index
        existing_similar_drug_ids = [id for id in similar_drug_ids if id in drug_data.index]
        if len(existing_similar_drug_ids) < len(similar_drug_ids):
             logging.warning(f"/get_alternative: Some similar drug IDs ({len(similar_drug_ids) - len(existing_similar_drug_ids)}) found by FAISS/mapping were not found in the main drug_data DataFrame index.")

        if not existing_similar_drug_ids:
             logging.warning("/get_alternative: No valid similar drug IDs found in the main data after filtering and index check.")
             # Prepare target drug dict, excluding embedding text
             target_drug_dict = target_drug_info.to_dict()
             if 'combined_embedding_text' in target_drug_dict:
                 del target_drug_dict['combined_embedding_text']
                 logging.debug("/get_alternative: Removed 'combined_embedding_text' from target_drug in response.")

             return jsonify({
                 'target_drug': target_drug_dict,
                 'similar_medicines': [], # Explicitly empty list
                 'best_alternative_recommendation': 'Similar drugs were identified by the search index, but their details could not be retrieved from the data.'
             }), 200


        similar_drugs_df = drug_data.loc[existing_similar_drug_ids]

        # Convert to list of dictionaries for LLM and response
        similar_meds_details = similar_drugs_df.to_dict(orient='records')

        # Prepare a summary list for the response (subset of columns)
        summary_cols = ['drug_id', 'name', 'generic_name', 'indication', 'mechanism', 'brand_name', 'synonyms'] # Added brand_name, synonyms
        # Filter for columns that actually exist in the DataFrame
        existing_summary_cols = [col for col in summary_cols if col in similar_drugs_df.columns]

        similar_meds_summary = similar_drugs_df[existing_summary_cols].to_dict(orient='records')

        logging.info(f"/get_alternative: Retrieved details for {len(similar_meds_details)} similar drugs.")


    except KeyError as e:
         logging.error(f"/get_alternative: KeyError retrieving details for similar drugs (missing drug_id in index?): {e}", exc_info=True) # Log traceback
         return jsonify({'error': 'Internal server error: Mismatch in drug IDs during detail retrieval.'}), 500
    except Exception as e:
         logging.error(f"/get_alternative: Error retrieving details for similar drugs: {e}", exc_info=True) # Log traceback
         return jsonify({'error': 'Internal server error retrieving similar drug details'}), 500

    # --- Step 4: Use LLM to determine the best alternative ---
    logging.info(f"/get_alternative: Querying LLM for best alternative among {len(similar_meds_details)} options.")
    target_drug_dict_for_llm = target_drug_info.to_dict() # Use the full dict for the LLM
    try:
        best_alternative_reasoning = get_best_alternative_from_llm(target_drug_dict_for_llm, similar_meds_details)
        logging.info("/get_alternative: LLM call completed.")
    except Exception as e:
         logging.error(f"/get_alternative: Error calling LLM service: {e}", exc_info=True) # Log traceback
         # Prepare target drug dict for response, excluding embedding text
         target_drug_dict_for_response = target_drug_info.to_dict()
         if 'combined_embedding_text' in target_drug_dict_for_response:
             del target_drug_dict_for_response['combined_embedding_text']
             logging.debug("/get_alternative: Removed 'combined_embedding_text' from target_drug in response due to LLM error.")

         return jsonify({
             'target_drug': target_drug_dict_for_response,
             'similar_medicines': similar_meds_summary,
             'best_alternative_recommendation': f'Could not generate a specific recommendation (LLM service error: {e})'
         }), 500 # Or 200 with an error message in the recommendation field


    # --- Step 5: Return the response ---
    logging.info("/get_alternative: Returning final response.")
    # Prepare target drug dict for response, excluding embedding text
    target_drug_dict_for_response = target_drug_info.to_dict()
    if 'combined_embedding_text' in target_drug_dict_for_response:
        del target_drug_dict_for_response['combined_embedding_text']
        logging.debug("/get_alternative: Removed 'combined_embedding_text' from target_drug in final response.")

    return jsonify({
        'target_drug': target_drug_dict_for_response, # Return richer info about target (without embedding text)
        'similar_medicines': similar_meds_summary, # Return summary list
        'best_alternative_recommendation': best_alternative_reasoning
    })


# --- New Route: Get Drug Details ---
@api_bp.route('/drug_details', methods=['GET'])
def get_drug_details_route():
    """
    API endpoint to get full details for a single drug, excluding embedding text.
    Searches by medicine name, generic name, brand name, or synonym.
    """
    medicine_name = request.args.get('name')
    if not medicine_name:
        logging.warning("Missing 'name' parameter in /drug_details request.")
        return jsonify({'error': 'Please provide a medicine name parameter (e.g., /api/drug_details?name=Aspirin)'}), 400

    # Check if drug_data is initialized
    if drug_data is None or drug_data.empty:
         logging.error("/drug_details: Data DataFrame is not initialized.")
         return jsonify({'error': 'Server not ready, data not loaded'}), 503 # Service Unavailable

    logging.info(f"Received /drug_details request for: {medicine_name}")

    try:
        # Use the helper function to find the drug
        target_drug_info = _find_drug_by_name(medicine_name)

        if target_drug_info is None:
            logging.warning(f"/drug_details: Medicine not found in dataset: {medicine_name}")
            return jsonify({'error': f"Medicine '{medicine_name}' not found."}), 404

        # Convert the Series to a dictionary
        drug_details_dict = target_drug_info.to_dict()

        # --- EXCLUDE combined_embedding_text ---
        if 'combined_embedding_text' in drug_details_dict:
            del drug_details_dict['combined_embedding_text']
            logging.debug("/drug_details: Removed 'combined_embedding_text' from response.")
        # --- End Exclusion ---


        # Return the modified dictionary as a JSON response
        logging.info(f"/drug_details: Found drug '{drug_details_dict.get('name', 'N/A')}' (ID: {drug_details_dict.get('drug_id', 'N/A')}). Returning details.")
        return jsonify(drug_details_dict), 200

    except Exception as e:
        logging.error(f"/drug_details: Error retrieving details for '{medicine_name}': {e}", exc_info=True)
        return jsonify({'error': f"An internal error occurred while retrieving details for '{medicine_name}'."}), 500


# --- New Route: Compare Medicines ---
@api_bp.route('/compare_medicines', methods=['GET'])
def compare_medicines_route():
    """
    API endpoint to compare the similarity of two medicines.
    Compares based on their embeddings.
    """
    name1 = request.args.get('name1')
    name2 = request.args.get('name2')

    if not name1 or not name2:
        logging.warning("Missing 'name1' or 'name2' parameter in /compare_medicines request.")
        return jsonify({'error': 'Please provide both name1 and name2 parameters (e.g., /api/compare_medicines?name1=Aspirin&name2=Ibuprofen)'}), 400

    # Check if resources are initialized
    required_cols = ['drug_id', 'name', 'combined_embedding_text'] # Need name and embedding text
    if drug_data is None or drug_data.empty or not all(col in drug_data.columns for col in required_cols) or sbert_model is None:
         logging.error("/compare_medicines: API resources not initialized properly or missing critical data columns.")
         logging.error(f"Data initialized: {drug_data is not None and not drug_data.empty}. Missing cols: {[col for col in required_cols if drug_data is not None and col not in drug_data.columns] if drug_data is not None else 'Data is None'}")
         logging.error(f"SBERT model initialized: {sbert_model is not None}")
         return jsonify({'error': 'Server not ready, resources not loaded or data incomplete'}), 503 # Service Unavailable

    logging.info(f"Received /compare_medicines request for: {name1} vs {name2}")

    try:
        # Find both drugs using the helper function
        drug1_info = _find_drug_by_name(name1)
        drug2_info = _find_drug_by_name(name2)

        if drug1_info is None:
            logging.warning(f"/compare_medicines: First medicine not found: {name1}")
            return jsonify({'error': f"Medicine '{name1}' not found."}), 404

        if drug2_info is None:
            logging.warning(f"/compare_medicines: Second medicine not found: {name2}")
            return jsonify({'error': f"Medicine '{name2}' not found."}), 404

        # Ensure both drugs have embedding text
        if 'combined_embedding_text' not in drug1_info or pd.isna(drug1_info['combined_embedding_text']) or drug1_info['combined_embedding_text'] == "":
             logging.error(f"/compare_medicines: Embedding text missing or empty for '{drug1_info['name']}' (ID: {drug1_info['drug_id']}).")
             return jsonify({'error': f"Embedding text missing for '{drug1_info['name']}'."}), 500

        if 'combined_embedding_text' not in drug2_info or pd.isna(drug2_info['combined_embedding_text']) or drug2_info['combined_embedding_text'] == "":
             logging.error(f"/compare_medicines: Embedding text missing or empty for '{drug2_info['name']}' (ID: {drug2_info['drug_id']}).")
             return jsonify({'error': f"Embedding text missing for '{drug2_info['name']}'."}), 500


        # Get the embedding texts
        text1 = drug1_info['combined_embedding_text']
        text2 = drug2_info['combined_embedding_text']

        # Generate embeddings for both texts
        logging.info(f"/compare_medicines: Generating embeddings for '{drug1_info['name']}' and '{drug2_info['name']}'.")
        embeddings = sbert_model.encode([text1, text2], convert_to_numpy=True)

        if embeddings is None or not isinstance(embeddings, np.ndarray) or embeddings.shape[0] != 2:
             raise ValueError("Failed to generate valid numpy embeddings for comparison.")

        embedding1 = embeddings[0].reshape(1, -1) # Reshape for cosine_similarity
        embedding2 = embeddings[1].reshape(1, -1) # Reshape for cosine_similarity


        # Calculate cosine similarity
        # cosine_similarity returns a 2D array [[score]], so we extract the score
        similarity_score = cosine_similarity(embedding1, embedding2)[0][0]

        logging.info(f"/compare_medicines: Calculated similarity between '{drug1_info['name']}' and '{drug2_info['name']}': {similarity_score:.4f}")

        # Return the result
        return jsonify({
            'medicine1': {'name': drug1_info['name'], 'id': drug1_info['drug_id']},
            'medicine2': {'name': drug2_info['name'], 'id': drug2_info['drug_id']},
            'similarity_percentage': float(similarity_score) * 100 # Ensure it's a standard float for JSON
        }), 200

    except Exception as e:
        logging.error(f"/compare_medicines: Error comparing '{name1}' and '{name2}': {e}", exc_info=True)
        return jsonify({'error': f"An internal error occurred while comparing '{name1}' and '{name2}'."}), 500

