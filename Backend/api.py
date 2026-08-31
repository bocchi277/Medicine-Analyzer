from functools import wraps
import threading
import time
from flask import Blueprint, request, jsonify
import logging
import numpy as np
from llm_service import get_best_alternative_from_llm
from vector_store import search_similar, get_stored_vector, cosine_similarity_of
import pandas as pd

# Ensure logging is configured to show INFO level or higher
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Global resources
faiss_index = None
faiss_id_map = None
# drug_id -> position in the FAISS index, used to read a drug's stored embedding
# instead of recomputing it with an embedding model at request time.
faiss_position_by_drug_id = None
drug_data = None
# Lowercase alias -> (drug_id, priority). Covers name, generic name, synonyms
# and brand names, so lookups are a dict hit rather than a scan of every row.
alias_index = {}

def init_api(index, id_map, position_by_drug_id, data_df):
    """Initialize API resources with preprocessed data."""
    global faiss_index, faiss_id_map, faiss_position_by_drug_id, drug_data, alias_index

    faiss_index = index
    faiss_id_map = id_map
    faiss_position_by_drug_id = position_by_drug_id
    alias_index = {}

    if data_df is None or data_df.empty:
        logging.error("Initialization failed: Empty DataFrame provided.")
        drug_data = pd.DataFrame() # Ensure drug_data is a DataFrame even if empty
        return

    # Take ownership of the frame rather than copying it: create_app hands this
    # over and keeps no reference, and a copy of 17k rows of drug text costs
    # ~50 MB of peak RSS during startup for no benefit.
    drug_data = data_df

    # The concatenated embedding text is only needed to *build* the index, which
    # now happens offline in build_index.py. It is the single largest column
    # (~23 MB of ~50 MB), so drop it once the index is loaded.
    if 'combined_embedding_text' in drug_data.columns:
        drug_data.drop(columns=['combined_embedding_text'], inplace=True)
        logging.info("Dropped 'combined_embedding_text' from runtime data (build-time only).")

    logging.info(f"API initializing with data shape: {drug_data.shape}")

    if 'drug_id' not in drug_data.columns:
        logging.error("Initialization failed: DataFrame must contain 'drug_id' column.")
        drug_data = pd.DataFrame()
        return

    if drug_data.index.name != 'drug_id':
        try:
            drug_data.set_index('drug_id', inplace=True, drop=False)
            logging.info("Set 'drug_id' as DataFrame index.")
        except Exception as e:
            logging.error(f"Error setting 'drug_id' as index: {e}")
            drug_data = pd.DataFrame()
            return

    alias_index = _build_alias_index(drug_data)

    logging.info(f"API initialized successfully with {len(drug_data)} entries.")
    logging.info(f"Search alias index built with {len(alias_index)} keys.")
    logging.info(f"Initialized drug_data columns: {drug_data.columns.tolist()}")


def _build_alias_index(df):
    """
    Build one lowercase name -> drug_id lookup covering every way a drug can be
    referred to, with ties resolved by how authoritative the source field is.

    This replaces scanning all 17k rows with str.contains() on every request.
    Beyond being O(1) instead of O(n), it fixes a correctness bug: the old code
    OR-ed an exact name match together with a *substring* match against the
    synonyms blob and then took whichever row happened to come first in the
    CSV. Searching "Ibuprofen" returned Dexibuprofen, because Dexibuprofen's
    synonyms contain the substring "ibuprofen" and its row is earlier.

    Multi-value fields use different separators in this dataset: generic_name
    and synonyms are newline-separated, brand_name is comma-separated.
    """
    # Lower number wins when two drugs claim the same alias.
    NAME, GENERIC, SYNONYM, BRAND = 1, 2, 3, 4
    sources = [
        ('name', NAME, None),
        ('generic_name', GENERIC, '\n'),
        ('synonyms', SYNONYM, '\n'),
        ('brand_name', BRAND, ','),
    ]

    index = {}

    def add(raw, drug_id, priority):
        key = raw.strip().lower()
        # Single characters are too ambiguous to be useful, and the placeholder
        # would otherwise map thousands of drugs onto one key.
        if len(key) < 2 or key == 'not available':
            return
        existing = index.get(key)
        if existing is None or priority < existing[1]:
            index[key] = (drug_id, priority)

    available = [(col, prio, sep) for col, prio, sep in sources if col in df.columns]
    for col, _, _ in sources:
        if col not in df.columns:
            logging.warning(f"Search column '{col}' not present; excluded from the alias index.")

    columns = {col: df[col].tolist() for col, _, _ in available}
    drug_ids = df.index.tolist()

    for row, drug_id in enumerate(drug_ids):
        for col, priority, separator in available:
            value = columns[col][row]
            if not value:
                continue
            if separator is None:
                add(value, drug_id, priority)
            else:
                for part in value.split(separator):
                    add(part, drug_id, priority)

    return index


def check_resources(*required_cols, require_vectors=False):
    """
    Decorator to validate API resources (data, index) and required data columns.

    Args:
        required_cols: data columns the route reads and cannot work without.
        require_vectors: set for routes that need the FAISS index and its
            position map (i.e. anything doing similarity work).
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
            elif require_vectors and (
                faiss_index is None
                or faiss_id_map is None
                or faiss_position_by_drug_id is None
            ):
                error = 'Search index not loaded. Service unavailable.'

            if error:
                logging.error(f"Resource check failed for route {request.path}: {error}")
                return jsonify({'error': error}), 503 # Service Unavailable
            return f(*args, **kwargs)
        return wrapper
    return decorator


# --- Rate limiting -----------------------------------------------------------
# The API is public and unauthenticated, and /get_alternative spends Gemini
# quota on every call. Without a cap, a single script can exhaust the quota or
# trip Render's outbound-traffic threshold and get the service suspended.
#
# A plain in-process counter is sufficient here precisely because the service
# runs a single worker with threads (see the Procfile). If it is ever scaled to
# multiple processes or instances, this becomes per-process and should be
# replaced with a shared store.
_rate_state = {}
_rate_lock = threading.Lock()
_RATE_MAX_ENTRIES = 4096


def _client_ip():
    """Best-effort client address, honouring Render's proxy header."""
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.remote_addr or 'unknown'


def rate_limit(max_calls, per_seconds):
    """Allow at most `max_calls` per `per_seconds` per client IP."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            key = (f.__name__, _client_ip())
            now = time.monotonic()
            cutoff = now - per_seconds

            with _rate_lock:
                # Opportunistically discard stale buckets so a long-running
                # process cannot accumulate an entry per address seen.
                if len(_rate_state) > _RATE_MAX_ENTRIES:
                    for stale_key in [
                        k for k, v in _rate_state.items() if not v or v[-1] < cutoff
                    ]:
                        _rate_state.pop(stale_key, None)

                hits = [t for t in _rate_state.get(key, []) if t > cutoff]
                if len(hits) >= max_calls:
                    retry_after = max(1, int(per_seconds - (now - hits[0])))
                    _rate_state[key] = hits
                    logging.warning(f"Rate limit hit for {key[1]} on {request.path}")
                    response = jsonify({
                        'error': 'Too many requests. Please wait a moment and try again.'
                    })
                    response.headers['Retry-After'] = str(retry_after)
                    return response, 429

                hits.append(now)
                _rate_state[key] = hits

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

    # Exclude internal/embedding columns. No lowercase helper columns exist any
    # more -- search runs off the alias index instead of duplicated columns.
    response.pop('combined_embedding_text', None)

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
    Resolve a user-supplied name to a single drug record.

    Resolution order:
      1. Exact match (case-insensitive) against the alias index, which prefers
         a drug's own name over its generic name, then synonyms, then brands.
      2. Failing that, a substring search over the same alias keys, preferring
         the shortest matching alias so "ibupro" lands on "ibuprofen" rather
         than on some longer unrelated string that happens to contain it.

    Args:
        medicine_name (str): The name to search for.

    Returns:
        pd.Series or None: The DataFrame row for the found drug, or None.
    """
    if drug_data is None or drug_data.empty:
        logging.error("find_drug called before drug_data is initialized.")
        return None
    if not medicine_name:
        return None

    query = medicine_name.strip().lower()
    if not query:
        return None

    match = alias_index.get(query)

    if match is None and len(query) >= 3:
        # Partial input. Rank candidates by, in order:
        #   1. whether the alias *starts with* the query -- a prefix match is
        #      what a user typing a partial name expects. Without this, "para"
        #      resolved to Cinacalcet via some interior substring rather than
        #      to Paracetamol.
        #   2. alias length, so the closest-fitting name wins.
        #   3. field priority, so a drug's own name beats a brand alias.
        # Gated at 3 characters: substring-matching one or two letters returns
        # an essentially arbitrary drug.
        best = None
        for alias, (drug_id, priority) in alias_index.items():
            if query in alias:
                candidate = (0 if alias.startswith(query) else 1,
                             len(alias), priority, drug_id)
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            match = (best[3], best[2])
            logging.info(f"find_drug: '{medicine_name}' matched by partial name -> {match[0]}")

    if match is None:
        return None

    drug_id = match[0]
    try:
        row = drug_data.loc[drug_id]
    except KeyError:
        logging.error(f"find_drug: alias resolved to drug_id {drug_id}, absent from the data.")
        return None

    # A duplicated index would make .loc return a DataFrame; take the first row.
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


@api_bp.route('/get_alternative', methods=['GET'])
# Needs the similarity index; the drug's own vector comes from there too.
@check_resources(require_vectors=True)
# This route calls Gemini, so it carries the tightest limit.
@rate_limit(max_calls=12, per_seconds=60)
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

        # --- Step 2: Look up the target's stored embedding and search FAISS ---
        # The target is by definition a row that was embedded at index build
        # time, so its vector is already in the index. Reading it back is exact
        # and avoids loading an embedding model into the web process.
        target_position = faiss_position_by_drug_id.get(target.name)
        if target_position is None:
            logging.error(
                f"/get_alternative: Drug ID {target.name} has no position in the search index. "
                "The index is stale relative to the data; rebuild with build_index.py."
            )
            return jsonify({'error': 'Internal server error: drug missing from search index.'}), 500

        logging.info(
            f"/get_alternative: Using stored vector at position {target_position} "
            f"for: {target.get('name', 'N/A')}"
        )
        embedding = get_stored_vector(faiss_index, target_position)
        if embedding is None:
            raise ValueError("Failed to read the stored embedding for the query drug.")

        # search_similar returns a list of dicts with 'drug_id' and 'similarity_score'
        logging.info("Calling search_similar with top_n=10")
        similar_indices_scores = search_similar(
            embedding,
            faiss_index,
            faiss_id_map,
            top_n=10,
            exclude_position=target_position,
        )

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
@rate_limit(max_calls=60, per_seconds=60)
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
# Needs the similarity index; both drugs' vectors are read from it.
@check_resources(require_vectors=True)
@rate_limit(max_calls=60, per_seconds=60)
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

        # Read both stored vectors out of the index rather than re-embedding.
        # Identical result: they were built from these exact rows and are unit
        # length, so their dot product *is* the cosine similarity.
        position1 = faiss_position_by_drug_id.get(drug1.name)
        position2 = faiss_position_by_drug_id.get(drug2.name)
        for drug, position in ((drug1, position1), (drug2, position2)):
            if position is None:
                logging.error(
                    f"/compare_medicines: '{drug.get('name', 'N/A')}' (ID: {drug.name}) has no "
                    "position in the search index. Rebuild with build_index.py."
                )
                return jsonify({
                    'error': f"'{drug.get('name', 'N/A')}' is missing from the search index."
                }), 500

        logging.info(
            f"/compare_medicines: Reading stored vectors for "
            f"'{drug1.get('name', 'N/A')}' and '{drug2.get('name', 'N/A')}'."
        )
        embedding1 = get_stored_vector(faiss_index, position1)
        embedding2 = get_stored_vector(faiss_index, position2)
        if embedding1 is None or embedding2 is None:
            raise ValueError("Failed to read stored embeddings for comparison.")

        similarity_score = cosine_similarity_of(embedding1, embedding2)

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

