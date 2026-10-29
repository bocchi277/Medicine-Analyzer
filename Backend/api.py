from bisect import bisect_left
from functools import wraps
import json
import os
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
# Sorted alias keys, for prefix search (autocomplete, partial names).
alias_keys = []
# drug_ids that are approved somewhere and not withdrawn.
approved_ids = set()
# drug_id -> number of brand names. A rough popularity signal: paracetamol has
# hundreds of brands, an obscure research compound has none.
brand_counts = {}

# Lower number wins when two drugs claim the same alias. CURATED is the
# hand-written Indian brand list, which must beat DrugBank: "Crocin" is a
# paracetamol brand in India but a saffron compound in DrugBank.
CURATED, NAME, GENERIC, SYNONYM, BRAND = 0, 1, 2, 3, 4
ALIAS_KIND = {CURATED: 'brand', NAME: 'name', GENERIC: 'generic', SYNONYM: 'synonym', BRAND: 'brand'}

# Partial names shorter than this are not guessed. "para" used to resolve to
# Paraoxon (a pesticide metabolite) because it was the shortest alias starting
# with "para". Short input now gets a "did you mean" list instead.
MIN_PARTIAL_LENGTH = 5

BRAND_ALIAS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brand_aliases_india.json')

DISCLAIMER = ('For information only. Similar drugs are not automatically safe substitutes. '
              'Ask a doctor or pharmacist before changing any medicine.')

def init_api(index, id_map, position_by_drug_id, data_df):
    """Initialize API resources with preprocessed data."""
    global faiss_index, faiss_id_map, faiss_position_by_drug_id, drug_data, alias_index
    global alias_keys, approved_ids, brand_counts

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
    alias_keys = sorted(alias_index)
    approved_ids = _approved_drug_ids(drug_data)
    if 'brand_name' in drug_data.columns:
        brand_counts = {
            drug_id: (0 if not isinstance(v, str) or v == 'Not Available' else v.count(',') + 1)
            for drug_id, v in zip(drug_data.index, drug_data['brand_name'])
        }
    logging.info(f"{len(approved_ids)} drugs are approved and not withdrawn.")

    logging.info(f"API initialized successfully with {len(drug_data)} entries.")
    logging.info(f"Search alias index built with {len(alias_index)} keys.")
    logging.info(f"Initialized drug_data columns: {drug_data.columns.tolist()}")


def _approved_drug_ids(df):
    """Drugs approved in at least one market and not withdrawn anywhere."""
    if 'groups' not in df.columns:
        return set(df.index)
    groups = df['groups'].fillna('').str.lower()
    keep = groups.str.contains('approved') & ~groups.str.contains('withdrawn')
    return set(df.index[keep])


def _load_curated_brands():
    try:
        with open(BRAND_ALIAS_FILE, encoding='utf-8') as f:
            return json.load(f).get('brands', {})
    except FileNotFoundError:
        logging.warning(f"{BRAND_ALIAS_FILE} not found; Indian brand names will not resolve.")
        return {}


def _build_alias_index(df):
    """
    Build one lowercase name -> (drug_id, priority, original text) lookup that
    covers every way a drug can be referred to.

    When two drugs claim the same alias, the more authoritative source wins:
    curated brand list, then the drug's own name, generic name, synonyms and
    DrugBank brand names. Without this, "Ibuprofen" resolved to Dexibuprofen,
    whose synonyms contain the word and whose row comes first in the CSV.

    Multi-value fields use different separators in this dataset: generic_name
    and synonyms are newline-separated, brand_name is comma-separated.
    """
    sources = [
        ('name', NAME, None),
        ('generic_name', GENERIC, '\n'),
        ('synonyms', SYNONYM, '\n'),
        ('brand_name', BRAND, ','),
    ]

    index = {}

    def add(raw, drug_id, priority):
        text = raw.strip()
        key = text.lower()
        # Single characters are too ambiguous to be useful, and the placeholder
        # would otherwise map thousands of drugs onto one key.
        if len(key) < 2 or key == 'not available':
            return
        existing = index.get(key)
        if existing is None or priority < existing[1]:
            index[key] = (drug_id, priority, text)

    available = [(col, prio, sep) for col, prio, sep in sources if col in df.columns]
    for col, _, _ in sources:
        if col not in df.columns:
            logging.warning(f"Search column '{col}' not present; excluded from the alias index.")

    columns = {col: df[col].tolist() for col, _, _ in available}
    drug_ids = df.index.tolist()

    for row, drug_id in enumerate(drug_ids):
        for col, priority, separator in available:
            value = columns[col][row]
            if not value or not isinstance(value, str):
                continue
            if separator is None:
                add(value, drug_id, priority)
            else:
                for part in value.split(separator):
                    add(part, drug_id, priority)

    # Curated brands point at a drug by its exact name.
    by_name = {}
    if 'name' in df.columns:
        for drug_id, name in zip(drug_ids, columns.get('name', df['name'].tolist())):
            by_name.setdefault(str(name).strip().lower(), drug_id)
    for brand, target in _load_curated_brands().items():
        drug_id = by_name.get(target.strip().lower())
        if drug_id is None:
            logging.warning(f"Curated brand '{brand}' points at unknown drug '{target}'; skipped.")
            continue
        add(brand.title() if brand.islower() else brand, drug_id, CURATED)

    return index


def _drug_name(drug_id):
    try:
        return str(drug_data.at[drug_id, 'name'])
    except (KeyError, ValueError):
        return str(drug_id)


def suggest(query, limit=8):
    """
    Names that start with (or, failing that, contain) the query.

    Ranked by: exact match, prefix before substring, approved drugs first,
    popular drugs (many brand names) first, then shorter aliases. One entry
    per drug. Popularity matters for short input: for "para" the user almost
    certainly wants Paracetamol, not Paramethadione.
    """
    q = (query or '').strip().lower()
    if len(q) < 2 or not alias_keys:
        return []

    candidates = []
    start = bisect_left(alias_keys, q)
    for key in alias_keys[start:start + 400]:
        if not key.startswith(q):
            break
        candidates.append((0, key))
    if len(candidates) < limit and len(q) >= 3:
        candidates += [(1, k) for k in alias_keys if q in k and not k.startswith(q)][:400]

    def rank(item):
        where, key = item
        drug_id, priority, _ = alias_index[key]
        popularity = min(brand_counts.get(drug_id, 0), 50) // 10   # 0..5, coarse on purpose
        return (key != q, where, drug_id not in approved_ids, -popularity,
                priority not in (NAME, CURATED), len(key), key)

    results, seen = [], set()
    for where, key in sorted(candidates, key=rank):
        drug_id, priority, text = alias_index[key]
        if drug_id in seen:
            continue
        seen.add(drug_id)
        name = _drug_name(drug_id)
        results.append({
            'label': text,
            'name': name,
            'drug_id': drug_id,
            'kind': ALIAS_KIND[priority],
            'approved': drug_id in approved_ids,
        })
        if len(results) >= limit:
            break
    return results


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
    """
    The caller's address, taken only from headers a client cannot forge.

    Render sits behind Cloudflare. Cloudflare overwrites CF-Connecting-IP with
    the address it actually received the connection from, so that header is
    the trustworthy one. The old code used the *first* X-Forwarded-For entry,
    which the client writes itself: sending a new fake value per request got
    around the rate limit. If CF-Connecting-IP is absent, the *last*
    X-Forwarded-For entry is used, because that one was added by our own proxy.
    """
    cf_ip = request.headers.get('CF-Connecting-IP', '').strip()
    if cf_ip:
        return cf_ip
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        last = forwarded.split(',')[-1].strip()
        if last:
            return last
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
    response['approved'] = drug_row.name in approved_ids

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

def resolve_drug(medicine_name):
    """
    Resolve a user-supplied name to a drug_id.

    Returns (drug_id or None, how) where how is 'exact', 'partial' or None.

      1. Exact match (case-insensitive) on any alias.
      2. For input of MIN_PARTIAL_LENGTH or more characters: the shortest
         alias that *starts with* the input ("ibupro" -> Ibuprofen). Interior
         substrings are not used for this, because "para" inside an unrelated
         name is not what anyone means.
      3. Otherwise nothing; callers show suggest() results instead.
    """
    if not medicine_name or not alias_index:
        return None, None
    query = medicine_name.strip().lower()
    if not query:
        return None, None

    match = alias_index.get(query)
    if match is not None:
        return match[0], 'exact'

    if len(query) >= MIN_PARTIAL_LENGTH:
        best = None
        start = bisect_left(alias_keys, query)
        for key in alias_keys[start:start + 400]:
            if not key.startswith(query):
                break
            drug_id, priority, _ = alias_index[key]
            candidate = (len(key), priority, key, drug_id)
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            logging.info(f"resolve_drug: '{medicine_name}' matched by prefix -> {best[3]}")
            return best[3], 'partial'

    return None, None


def find_drug(medicine_name: str):
    """Return the DataFrame row for a name (see resolve_drug), or None."""
    if drug_data is None or drug_data.empty:
        logging.error("find_drug called before drug_data is initialized.")
        return None

    drug_id, _ = resolve_drug(medicine_name)
    if drug_id is None:
        return None
    try:
        row = drug_data.loc[drug_id]
    except KeyError:
        logging.error(f"find_drug: alias resolved to drug_id {drug_id}, absent from the data.")
        return None

    # A duplicated index would make .loc return a DataFrame; take the first row.
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def not_found_response(medicine_name, noun='Medicine'):
    """404 body with "did you mean" suggestions the page can show as buttons."""
    suggestions = suggest(medicine_name, limit=6)
    message = f"{noun} '{medicine_name}' not found."
    if suggestions:
        message += ' Did you mean one of the suggestions below?'
    else:
        message += ' Please check the spelling or try a different name.'
    return jsonify({'error': message, 'suggestions': suggestions}), 404


def _flag(name, default):
    value = request.args.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ('0', 'false', 'no', 'off')


@api_bp.route('/suggest', methods=['GET'])
@check_resources()
@rate_limit(max_calls=240, per_seconds=60)
def suggest_route():
    """Autocomplete: GET /api/suggest?q=ibu -> {"suggestions": [...]}."""
    query = (request.args.get('q') or '')[:80]
    return jsonify({'suggestions': suggest(query)}), 200


# How many alternatives the page shows, and how far down the neighbour list to
# look for them once unapproved drugs are filtered out.
TOP_N = 10
SEARCH_DEPTH = 80


@api_bp.route('/get_alternative', methods=['GET'])
# Needs the similarity index; the drug's own vector comes from there too.
@check_resources(require_vectors=True)
# This route can call Gemini, so it carries the tightest limit.
@rate_limit(max_calls=12, per_seconds=60)
def get_alternative_route():
    """
    GET /api/get_alternative?name=Ibuprofen[&approved_only=1][&llm=1]

    Finds the drug, takes its stored vector, returns the nearest drugs by
    cosine similarity and (if llm=1) asks Gemini to pick the best one.

    approved_only (default on): skip experimental, investigational-only and
    withdrawn drugs. A patient-facing "alternative" that was never approved,
    or was withdrawn, is not useful and can be harmful.
    llm (default on): the details page sets llm=0 because it only needs the
    list, so it does not spend Gemini quota a second time.
    """
    medicine_name = request.args.get('name')
    if not medicine_name or not medicine_name.strip():
        logging.warning("Missing 'name' parameter in /get_alternative request.")
        return jsonify({'error': 'Please provide a medicine name parameter'}), 400
    medicine_name = medicine_name.strip()[:120]
    approved_only = _flag('approved_only', True)
    use_llm = _flag('llm', True)

    logging.info(f"/get_alternative: '{medicine_name}' approved_only={approved_only} llm={use_llm}")

    try:
        target = find_drug(medicine_name)
        if target is None:
            return not_found_response(medicine_name)

        # The target was embedded at build time, so its vector is already in
        # the index. Reading it back avoids loading an embedding model here.
        target_position = faiss_position_by_drug_id.get(target.name)
        if target_position is None:
            logging.error(
                f"/get_alternative: Drug ID {target.name} has no position in the search index. "
                "The index is stale relative to the data; rebuild with build_index.py."
            )
            return jsonify({'error': 'Internal server error: drug missing from search index.'}), 500

        embedding = get_stored_vector(faiss_index, target_position)
        if embedding is None:
            raise ValueError("Failed to read the stored embedding for the query drug.")

        neighbours = search_similar(
            embedding, faiss_index, faiss_id_map,
            top_n=SEARCH_DEPTH if approved_only else TOP_N,
            exclude_position=target_position,
        )

        similar_meds = []
        skipped = 0
        for hit in neighbours:
            drug_id = hit.get('drug_id')
            if drug_id is None or drug_id == target.name or drug_id not in drug_data.index:
                continue
            if approved_only and drug_id not in approved_ids:
                skipped += 1
                continue
            row = drug_data.loc[drug_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            record = prepare_drug_response(row)
            record['similarity_percentage'] = round(float(hit['similarity_score']) * 100, 2)
            similar_meds.append(record)
            if len(similar_meds) >= TOP_N:
                break

        target_record = prepare_drug_response(target)
        body = {
            'target': target_record,
            'similar': similar_meds,
            'filters': {'approved_only': approved_only, 'skipped_unapproved': skipped},
            'disclaimer': DISCLAIMER,
        }

        if not similar_meds:
            body['recommendation'] = ('No approved drugs close to this one were found in the index.'
                                      if approved_only else
                                      'No other similar drugs were found in the index.')
            return jsonify(body), 200

        if not use_llm:
            body['recommendation'] = None
            return jsonify(body), 200

        try:
            # The LLM gets the same records the page shows, minus the score.
            candidates = [{k: v for k, v in d.items() if k != 'similarity_percentage'}
                          for d in similar_meds]
            body['recommendation'] = get_best_alternative_from_llm(target_record, candidates)
        except Exception as e:
            logging.error(f"/get_alternative: Error calling LLM service: {e}", exc_info=True)
            # The list is still useful without the AI pick.
            body['recommendation'] = (f'Could not generate a specific recommendation '
                                      f'(LLM service error: {type(e).__name__}).')
        return jsonify(body), 200

    except Exception as e:
        logging.error(f"Error in /get_alternative route: {str(e)}", exc_info=True)
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
            return not_found_response(medicine_name, noun='Drug')

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
            response, status = not_found_response(missing_drug)
            payload = response.get_json()
            payload['missing'] = 'name1' if drug1 is None else 'name2'
            return jsonify(payload), status

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

