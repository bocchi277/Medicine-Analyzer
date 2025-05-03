from functools import wraps
from flask import Blueprint, request, jsonify
import logging
import numpy as np
from llm_service import get_best_alternative_from_llm
from vector_store import search_similar
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Ensure logging is configured to show INFO level or higher
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Global resources
sbert_model = None
faiss_index = None
faiss_id_map = None
drug_data = None
# Store preprocessed lowercased column names for efficient searching
search_columns = {}

def init_api(model, index, id_map, data_df):
    """Initialize API resources with preprocessed data."""
    global sbert_model, faiss_index, faiss_id_map, drug_data, search_columns

    sbert_model = model
    faiss_index = index
    faiss_id_map = id_map
    search_columns.clear() # Clear previous search columns if re-initializing

    if data_df is None or data_df.empty:
        logging.error("Initialization failed: Empty DataFrame provided.")
        drug_data = pd.DataFrame() # Ensure drug_data is a DataFrame even if empty
        return

    drug_data = data_df.copy() # Work on a copy to avoid modifying the original DataFrame passed in
    logging.info(f"API initializing with data shape: {drug_data.shape}")

    if 'drug_id' not in drug_data.columns:
        logging.error("Initialization failed: DataFrame must contain 'drug_id' column.")
        # Decide if you want to raise an error or just log and return empty data
        drug_data = pd.DataFrame() # Reset to empty if critical column is missing
        return # Stop initialization if drug_id is missing

    # Set drug_id as index for efficient lookup using .loc
    try:
        drug_data.set_index('drug_id', inplace=True, drop=False) # Keep drug_id as a column too
        logging.info("Set 'drug_id' as DataFrame index.")
    except Exception as e:
        logging.error(f"Error setting 'drug_id' as index: {e}")
        # Handle potential issues if drug_id is not unique or has problematic values
        drug_data = pd.DataFrame() # Reset if index setting fails critically
        return


    # Preprocess search columns by creating lowercased versions
    search_columns['main'] = []
    cols_to_lowercase = ['name', 'generic_name', 'brand_name']
    for col in cols_to_lowercase:
        if col in drug_data.columns:
            # Ensure column is string type before lowercasing
            if pd.api.types.is_string_dtype(drug_data[col]):
                 drug_data[f'{col}_lower'] = drug_data[col].str.lower().fillna("") # Fill NaN with empty string for safety
                 search_columns['main'].append(f'{col}_lower')
                 logging.debug(f"Created lowercased column: {col}_lower")
            else:
                 logging.warning(f"Column '{col}' is not string dtype ({drug_data[col].dtype}), skipping lowercasing for search.")

    if 'synonyms' in drug_data.columns:
        if pd.api.types.is_string_dtype(drug_data['synonyms']):
             drug_data['synonyms_lower'] = drug_data['synonyms'].str.lower().fillna("") # Fill NaN with empty string
             search_columns['synonyms'] = 'synonyms_lower'
             logging.debug("Created lowercased column: synonyms_lower")
        else:
             logging.warning("Column 'synonyms' is not string dtype, skipping lowercasing for search.")

    # --- DEBUG LOGGING: Check for smiles column during init ---
    if 'smiles' in drug_data.columns:
        logging.info("API initialized: 'smiles' column FOUND in drug_data.")
    else:
        logging.warning("API initialized: 'smiles' column NOT FOUND in drug_data. Smiles will not be available in responses.")
    # --- END DEBUG LOGGING ---


    logging.info(f"API initialized successfully with {len(drug_data)} entries.")
    logging.info(f"Available search columns: {search_columns}")
    # Log columns in the initialized data
    logging.info(f"Initialized drug_data columns: {drug_data.columns.tolist()}")


def check_resources(*required_cols):
    """
    Decorator to validate API resources (data, models) and required data columns.
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            error = None
            if drug_data is None or drug_data.empty:
                error = 'Drug data not loaded or is empty. Service unavailable.'
            elif not all(col in drug_data.columns for col in required_cols):
                missing = [col for col in required_cols if col not in drug_data.columns]
                error = f'Required data columns missing: {missing}. Service unavailable.'
            # Check only models needed for all routes decorated with check_resources
            # If a route needs a specific model (like sbert), check within the route or use a more specific decorator
            # For now, let's assume sbert is always needed if combined_embedding_text is required
            if 'combined_embedding_text' in required_cols and sbert_model is None:
                 error = 'SBERT model not loaded. Service unavailable.'
            # Add checks for faiss_index and faiss_id_map if needed by the route
            if 'combined_embedding_text' in required_cols and (faiss_index is None or faiss_id_map is None):
                 error = 'FAISS index or ID map not loaded. Service unavailable.'


            if error:
                logging.error(f"Resource check failed for route {request.path}: {error}")
                return jsonify({'error': error}), 503 # Service Unavailable
            return f(*args, **kwargs)
        return wrapper
    return decorator

def prepare_drug_response(drug_row: pd.Series):
    """
    Prepare a drug record (Pandas Series row) for API response.
    Excludes internal/embedding columns and includes smiles if available.
    NOTE: This function assumes the input drug_row Series contains the
          columns needed for the response, including 'smiles' if it
          is expected to be returned.
    """
    if not isinstance(drug_row, pd.Series):
        logging.error(f"prepare_drug_response received invalid type: {type(drug_row)}")
        return {} # Return empty dict for invalid input

    # Start with the full dictionary representation of the row
    response = drug_row.to_dict()

    # Exclude internal/embedding columns
    response.pop('combined_embedding_text', None)
    # Exclude lowercased search columns
    for col in search_columns.get('main', []):
        response.pop(col, None)
    if 'synonyms' in search_columns:
         response.pop(search_columns['synonyms'], None)

    # --- Explicitly handle smiles ---
    # Check if 'smiles' column exists in the original Series AND its value is not NaN
    if 'smiles' in drug_row.index and pd.notna(drug_row.loc['smiles']):
         response['smiles'] = drug_row.loc['smiles'] # Add smiles to the response dictionary
         # Ensure it wasn't accidentally removed by a pop if its name was similar
         # (Unlikely, but defensive)
    else:
         # If smiles is NaN or not in the original row index, ensure it's not in the response
         response.pop('smiles', None) # Remove if it somehow got in or was NaN


    return response

def find_drug(medicine_name: str):
    """
    Efficient drug lookup in the global drug_data DataFrame
    using preprocessed lowercased columns.

    Args:
        medicine_name (str): The name to search for.

    Returns:
        pd.Series or None: The DataFrame row for the found drug, or None if not found.
    """
    if drug_data is None or drug_data.empty:
        logging.error("find_drug called before drug_data is initialized.")
        return None

    name_lower = medicine_name.lower()
    # Initialize mask with False for all rows
    mask = pd.Series(False, index=drug_data.index)

    # Search main columns (name, generic_name, brand_name)
    for col_lower in search_columns.get('main', []):
         # Ensure the lowercased column exists before using it
         if col_lower in drug_data.columns:
              mask |= (drug_data[col_lower] == name_lower)
         else:
              logging.warning(f"Lowercase search column '{col_lower}' not found in drug_data.")


    # Search synonyms column
    syn_col_lower = search_columns.get('synonyms')
    if syn_col_lower and syn_col_lower in drug_data.columns:
         # Use .str.contains with na=False to handle empty/NaN synonyms safely
         mask |= drug_data[syn_col_lower].str.contains(name_lower, na=False)
    elif syn_col_lower:
         logging.warning(f"Lowercase synonyms column '{syn_col_lower}' not found in drug_data.")


    # Filter the DataFrame using the mask
    results = drug_data[mask]

    # Return the first matching row if any results are found
    return results.iloc[0] if not results.empty else None


@api_bp.route('/get_alternative', methods=['GET'])
# Require combined_embedding_text as it's needed for embedding and search
@check_resources('combined_embedding_text')
def get_alternative_route():
    """
    API endpoint to find similar drugs and the best alternative.
    Allows searching by medicine name, generic name, brand name, or synonym.
    Includes similarity percentage in the response and smiles for the target drug.
    """
    medicine_name = request.args.get('name')
    if not medicine_name:
        logging.warning("Missing 'name' parameter in /get_alternative request.")
        return jsonify({'error': 'Please provide a medicine name parameter'}), 400

    logging.info(f"Received /get_alternative request for: {medicine_name}")

    try:
        # --- Step 1: Find the target drug ---
        target = find_drug(medicine_name)
        if target is None:
            logging.warning(f"/get_alternative: Medicine not found in dataset: {medicine_name}")
            return jsonify({'error': f"Medicine '{medicine_name}' not found. Please check the spelling or try a different name."}), 404

        # Ensure target drug has embedding text
        if pd.isna(target.get('combined_embedding_text')) or target.get('combined_embedding_text') == "":
             logging.error(f"/get_alternative: Embedding text missing or empty for target drug ID {target.name}. Check data processing.")
             return jsonify({'error': 'Internal server error: Embedding text missing for target drug.'}), 500

        # --- Step 2: Generate embedding and search FAISS ---
        logging.info(f"/get_alternative: Generating embedding and searching FAISS for: {target.get('name', 'N/A')}")
        # Ensure sbert_model is available (checked by decorator, but defensive check)
        if sbert_model is None:
             logging.error("/get_alternative: SBERT model is unexpectedly None.")
             return jsonify({'error': 'Internal server error: Embedding model not loaded.'}), 503 # Should be caught by decorator

        embedding = sbert_model.encode(target.combined_embedding_text, convert_to_numpy=True)
        if embedding is None or not isinstance(embedding, np.ndarray) or embedding.size == 0:
             raise ValueError("Failed to generate a valid numpy embedding for the query drug.")
        logging.debug(f"/get_alternative: Embedding generated with shape: {embedding.shape}")

        # Ensure FAISS resources are available (checked by decorator)
        if faiss_index is None or faiss_id_map is None:
             logging.error("/get_alternative: FAISS resources are unexpectedly None.")
             return jsonify({'error': 'Internal server error: Search index not loaded.'}), 503 # Should be caught by decorator


        # search_similar function needs to return a list of dicts with 'drug_id' and 'similarity_score'
        logging.info("Calling search_similar with top_n=10")
        similar_indices_scores = search_similar(embedding, faiss_index, faiss_id_map, top_n=10)

        logging.info(f"/get_alternative: search_similar returned {len(similar_indices_scores)} results.")
        if similar_indices_scores:
             logging.debug(f"/get_alternative: Top 5 results from search_similar: {similar_indices_scores[:5]}")
        else:
             logging.debug("/get_alternative: search_similar returned an empty list.")


        # --- Step 3: Process search results, filter target, retrieve details, add scores ---
        # Create a dictionary mapping drug_id to its similarity score for easy lookup
        similar_drugs_with_scores = {
            res['drug_id']: res['similarity_score']
            for res in similar_indices_scores
            # Filter out the target drug itself. Use target.name (which is the index/drug_id)
            if 'drug_id' in res and res['drug_id'] != target.name
        }

        similar_drug_ids = list(similar_drugs_with_scores.keys())

        logging.info(f"/get_alternative: After filtering target drug (ID: {target.name}), {len(similar_drug_ids)} similar drug IDs remain.")
        if similar_drug_ids:
             logging.debug(f"/get_alternative: Filtered similar drug IDs with scores: {similar_drugs_with_scores}")


        if not similar_drug_ids:
             logging.warning(f"/get_alternative: No distinct similar drugs found for {target.get('name', 'N/A')} after filtering.")
             return jsonify({
                 'target': prepare_drug_response(target), # Use prepare_drug_response for the target
                 'similar': [], # Explicitly empty list
                 'recommendation': 'The search found the target medicine but no other distinct similar drugs in the index within the top results.'
             }), 200


        # Retrieve full data for the similar drugs using their IDs
        # Use .loc for index-based lookup, ensuring IDs exist in the index
        existing_similar_drug_ids = [id for id in similar_drug_ids if id in drug_data.index]
        if len(existing_similar_drug_ids) < len(similar_drug_ids):
             logging.warning(f"/get_alternative: Some similar drug IDs ({len(similar_drug_ids) - len(existing_similar_drug_ids)}) found by FAISS/mapping were not found in the main drug_data DataFrame index.")

        if not existing_similar_drug_ids:
             logging.warning("/get_alternative: No valid similar drug IDs found in the main data after filtering and index check.")
             return jsonify({
                 'target': prepare_drug_response(target), # Use prepare_drug_response for the target
                 'similar': [], # Explicitly empty list
                 'recommendation': 'Similar drugs were identified by the search index, but their details could not be retrieved from the data.'
             }), 200

        # Retrieve the DataFrame rows for the existing similar drug IDs
        similar_df = drug_data.loc[existing_similar_drug_ids]

        # Prepare lists for LLM and response
        similar_meds_details = []
        similar_meds_summary = []

        # Iterate through the similar_drug_ids (which have corresponding scores)
        for drug_id in similar_drug_ids:
            # Ensure the drug_id exists in the retrieved similar_df (it should if it's in existing_similar_drug_ids)
            if drug_id in similar_df.index:
                row = similar_df.loc[drug_id] # Get the row using .loc with the drug_id
                similarity_score = similar_drugs_with_scores[drug_id] # Get the score from the scores dict
                similarity_percentage = round(float(similarity_score) * 100, 2) # Calculate percentage

                # Prepare full details for LLM (using prepare_drug_response to exclude embedding text etc.)
                # LLM might benefit from smiles, so let prepare_drug_response handle its inclusion
                full_details = prepare_drug_response(row)
                similar_meds_details.append(full_details)

                # Prepare summary details for response (using prepare_drug_response and adding percentage)
                summary_dict = prepare_drug_response(row)
                summary_dict['similarity_percentage'] = similarity_percentage
                # If you want smiles in the similar summary list, uncomment below:
                # if 'smiles' in row.index and pd.notna(row.loc['smiles']):
                #      summary_dict['smiles'] = row.loc['smiles']
                similar_meds_summary.append(summary_dict)
            else:
                logging.warning(f"Drug ID {drug_id} found in similar_drug_ids but not in retrieved similar_df. Skipping.")


        logging.info(f"/get_alternative: Prepared details and added similarity scores for {len(similar_meds_summary)} similar drugs.")


        # --- Step 4: Use LLM to determine the best alternative ---
        logging.info(f"/get_alternative: Querying LLM for best alternative among {len(similar_meds_details)} options.")
        # Pass the target drug details and the list of similar drug details to the LLM service
        # Use prepare_drug_response for the target drug details sent to the LLM
        target_drug_dict_for_llm = prepare_drug_response(target)

        try:
            recommendation = get_best_alternative_from_llm(target_drug_dict_for_llm, similar_meds_details)
            logging.info("/get_alternative: LLM call completed.")
        except Exception as e:
            logging.error(f"/get_alternative: Error calling LLM service: {e}", exc_info=True)
            # Return partial result (similar drugs) with an error message in the recommendation field
            return jsonify({
                'target': prepare_drug_response(target), # Use prepare_drug_response for the target in the response
                'similar': similar_meds_summary, # Return the summary list with percentages
                'recommendation': f'Could not generate a specific recommendation (LLM service error: {type(e).__name__}).'
            }), 500 # Or 200 with an error message


        # --- Step 5: Return the final response ---
        logging.info("/get_alternative: Returning final response.")
        return jsonify({
            'target': prepare_drug_response(target), # Use prepare_drug_response for the target in the response (includes smiles if available)
            'similar': similar_meds_summary, # Return the summary list with percentages
            'recommendation': recommendation
        })

    except Exception as e:
        logging.error(f"Error in /get_alternative route: {str(e)}", exc_info=True)
        # Catch any unexpected errors and return a generic 500
        return jsonify({'error': 'Internal server error processing request.'}), 500

@api_bp.route('/drug_details', methods=['GET'])
# No specific columns required other than those needed for lookup (handled by find_drug)
@check_resources()
def get_drug_details_route():
    """
    API endpoint to get full details for a single drug, excluding embedding text.
    Searches by medicine name, generic name, brand name, or synonym.
    Includes smiles if available.
    """
    medicine_name = request.args.get('name')
    if not medicine_name:
        logging.warning("Missing 'name' parameter in /drug_details request.")
        return jsonify({'error': 'Please provide a medicine name parameter'}), 400

    logging.info(f"Received /drug_details request for: {medicine_name}")

    try:
        drug = find_drug(medicine_name)
        if drug is None:
             logging.warning(f"/drug_details: Drug not found: {medicine_name}")
             return jsonify({'error': f"Drug '{medicine_name}' not found."}), 404

        # Use prepare_drug_response to format the output (includes smiles if available)
        response_data = prepare_drug_response(drug)
        logging.info(f"/drug_details: Found drug '{response_data.get('name', 'N/A')}' (ID: {response_data.get('drug_id', 'N/A')}). Returning details.")
        return jsonify(response_data), 200

    except Exception as e:
        logging.error(f"Error in /drug_details route for '{medicine_name}': {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error retrieving drug details.'}), 500

@api_bp.route('/compare_medicines', methods=['GET'])
# Require combined_embedding_text for generating embeddings for comparison
@check_resources('combined_embedding_text')
def compare_medicines_route():
    """
    API endpoint to compare the similarity of two medicines.
    Compares based on their embeddings.
    Includes smiles for both drugs if available.
    """
    name1 = request.args.get('name1')
    name2 = request.args.get('name2')
    if not name1 or not name2:
        logging.warning("Missing 'name1' or 'name2' parameter in /compare_medicines request.")
        return jsonify({'error': 'Please provide both name1 and name2 parameters'}), 400

    logging.info(f"Received /compare_medicines request for: {name1} vs {name2}")

    try:
        drug1 = find_drug(name1)
        drug2 = find_drug(name2)
        if drug1 is None or drug2 is None:
            missing_drug = name1 if drug1 is None else name2
            logging.warning(f"/compare_medicines: One or both drugs not found: {name1}, {name2}")
            return jsonify({'error': f"Medicine '{missing_drug}' not found."}), 404

        # Ensure both drugs have embedding text
        if pd.isna(drug1.get('combined_embedding_text')) or drug1.get('combined_embedding_text') == "":
             logging.error(f"/compare_medicines: Embedding text missing or empty for '{drug1.get('name', 'N/A')}' (ID: {drug1.name}).")
             return jsonify({'error': f"Embedding text missing for '{drug1.get('name', 'N/A')}'."}), 500

        if pd.isna(drug2.get('combined_embedding_text')) or drug2.get('combined_embedding_text') == "":
             logging.error(f"/compare_medicines: Embedding text missing or empty for '{drug2.get('name', 'N/A')}' (ID: {drug2.name}).")
             return jsonify({'error': f"Embedding text missing for '{drug2.get('name', 'N/A')}'."}), 500


        # Get the embedding texts
        text1 = drug1.combined_embedding_text
        text2 = drug2.combined_embedding_text

        # Generate embeddings for both texts
        logging.info(f"/compare_medicines: Generating embeddings for '{drug1.get('name', 'N/A')}' and '{drug2.get('name', 'N/A')}'.")
        # Ensure sbert_model is available (checked by decorator)
        if sbert_model is None:
             logging.error("/compare_medicines: SBERT model is unexpectedly None.")
             return jsonify({'error': 'Internal server error: Embedding model not loaded.'}), 503 # Should be caught by decorator

        embeddings = sbert_model.encode([text1, text2], convert_to_numpy=True)

        if embeddings is None or not isinstance(embeddings, np.ndarray) or embeddings.shape[0] != 2 or embeddings.size == 0:
             raise ValueError("Failed to generate valid numpy embeddings for comparison.")

        embedding1 = embeddings[0].reshape(1, -1) # Reshape for cosine_similarity
        embedding2 = embeddings[1].reshape(1, -1) # Reshape for cosine_similarity


        # Calculate cosine similarity
        # cosine_similarity returns a 2D array [[score]], so we extract the score
        similarity_score = cosine_similarity(embedding1, embedding2)[0][0]

        logging.info(f"/compare_medicines: Calculated similarity between '{drug1.get('name', 'N/A')}' and '{drug2.get('name', 'N/A')}': {similarity_score:.4f}")

        # Return the result
        return jsonify({
            'drug1': prepare_drug_response(drug1), # Use prepare_drug_response (includes smiles if available)
            'drug2': prepare_drug_response(drug2), # Use prepare_drug_response (includes smiles if available)
            'similarity_score': float(similarity_score), # Return the raw score
            'similarity_percentage': round(float(similarity_score) * 100, 2) # Return the percentage
        }), 200

    except Exception as e:
        logging.error(f"Error in /compare_medicines route for '{name1}' vs '{name2}': {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error comparing medicines.'}), 500

