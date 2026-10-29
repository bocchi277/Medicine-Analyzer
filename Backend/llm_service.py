"""
Gemini-backed "best alternative" recommendation.

Uses the current `google-genai` SDK. The old `google-generativeai` package has
reached end of support and pulled in grpc, which cost ~55 MB of resident memory
versus ~26 MB here -- material when the whole service has a 512 MB budget.

Both the SDK import and the client are created lazily on first use, so requests
that never need an LLM (/drug_details, /compare_medicines) never pay for it.
"""

import logging
import threading
from collections import OrderedDict

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_client = None
_client_error = None
_client_lock = threading.Lock()


def _get_client():
    """
    Build (once) and return the Gemini client.

    Returns (client, error_message). Exactly one is non-None.
    """
    global _client, _client_error

    if _client is not None or _client_error is not None:
        return _client, _client_error

    with _client_lock:
        # Re-check inside the lock: another thread may have initialised already.
        if _client is not None or _client_error is not None:
            return _client, _client_error

        if not config.GEMINI_API_KEY:
            _client_error = "LLM service not configured: GEMINI_API_KEY is not set."
            logging.error(_client_error)
            return None, _client_error

        try:
            # Imported here rather than at module scope so the ~26 MB cost is
            # only paid by processes that actually make a recommendation.
            from google import genai

            _client = genai.Client(api_key=config.GEMINI_API_KEY)
            logging.info(f"Gemini client initialised (model: {config.GEMINI_MODEL_NAME}).")
        except Exception as e:
            _client_error = f"Failed to initialise the Gemini client ({type(e).__name__})."
            logging.exception("Gemini client initialisation failed.")

    return _client, _client_error


# Long DrugBank fields (mechanism, pharmacodynamics) can run to thousands of
# characters. The model only needs the gist to compare drugs, and every extra
# token costs latency and quota, so each field is clipped.
_FIELD_LIMIT = 400

_MISSING = {"", "not available", "n/a", "none"}


def _field(drug, key):
    value = drug.get(key)
    if value is None:
        return "Not listed"
    text = " ".join(str(value).split())
    if text.lower() in _MISSING:
        return "Not listed"
    if len(text) > _FIELD_LIMIT:
        text = text[:_FIELD_LIMIT].rsplit(" ", 1)[0] + " ..."
    return text


def format_drug_info_for_prompt(drug_details):
    """Format one drug record for the prompt, using only fields from the data."""
    if not isinstance(drug_details, dict):
        logging.warning(f"Expected dict for drug_details, but got {type(drug_details)}")
        return "Invalid drug details format."

    fields = [
        ("Name", "name"),
        ("Approval status", "groups"),
        ("Indication", "indication"),
        ("Mechanism", "mechanism"),
        ("Pharmacodynamics", "pharmacodynamics"),
        ("Metabolism", "metabolism"),
        ("Brand names", "brand_name"),
    ]
    lines = [f"- {fields[0][0]}: {_field(drug_details, fields[0][1])}"]
    lines += [f"  {label}: {_field(drug_details, key)}" for label, key in fields[1:]]
    return "\n".join(lines) + "\n"


# Bump when the prompt changes so cached answers from the old prompt are not
# served for the new one.
PROMPT_VERSION = 2


def _build_prompt(target_drug_details, similar_drugs_details):
    target_name = _field(target_drug_details, "name")
    prompt = (
        "You help a user explore possible alternatives to a medicine. Use ONLY the "
        "DrugBank records below. Do not add facts from outside these records.\n\n"
        f"**Target drug:**\n{format_drug_info_for_prompt(target_drug_details)}\n"
        "**Candidates:**\n"
    )
    for i, drug in enumerate(similar_drugs_details):
        if isinstance(drug, dict):
            prompt += f"{i + 1}. {format_drug_info_for_prompt(drug)}\n"
        else:
            logging.warning(f"Skipping invalid item in similar_drugs_details at index {i}: {type(drug)}")

    prompt += (
        "**Task:**\n"
        f"1. Pick the single candidate that is the closest substitute for {target_name}. "
        "A shared indication matters most, then a similar mechanism. Prefer approved drugs.\n"
        "2. If no candidate has a clearly similar use, answer \"No suitable alternative "
        "in this list\" as the Best Alternative and explain why in the reasoning.\n"
        "3. Brand names: copy them only from the chosen drug's \"Brand names\" field. "
        "If that field says \"Not listed\", write \"None listed in the data\". "
        "Never invent brand names.\n\n"
        "Reply in exactly this format, plain text, no markdown:\n"
        "Best Alternative: <candidate name>\n"
        "Reasoning: <one or two sentences that cite the data above>\n"
        "Brand names: <from the data, or None listed in the data>\n"
    )
    return prompt


# --- Answer cache ------------------------------------------------------------
# The same target always yields the same candidates (the index is static), so
# the same question reaches Gemini again and again. Caching successful answers
# removes repeat calls: faster replies and far less quota used. Bounded LRU so
# memory stays flat; per process, which is fine with one worker.
_CACHE_MAX = 1024
_cache = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(target_drug_details, similar_drugs_details):
    ids = tuple(str(d.get("drug_id", d.get("name", ""))) for d in similar_drugs_details
                if isinstance(d, dict))
    return (str(target_drug_details.get("drug_id", target_drug_details.get("name", ""))),
            ids, config.GEMINI_MODEL_NAME, PROMPT_VERSION)


def _cache_get(key):
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    return None


def _cache_put(key, value):
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


def clear_cache():
    with _cache_lock:
        _cache.clear()


def get_best_alternative_from_llm(target_drug_details, similar_drugs_details):
    """
    Queries the Gemini API to suggest the best alternative.

    Args:
        target_drug_details (dict): Dictionary with details of the original drug.
        similar_drugs_details (list[dict]): List of dictionaries with details of similar drugs.

    Returns:
        str: The LLM's recommendation text or a human-readable error message.
    """
    if not similar_drugs_details:
        logging.info("No similar drugs provided to LLM for evaluation.")
        return "No similar drugs found to evaluate."

    if not isinstance(target_drug_details, dict):
        logging.error(f"Expected dict for target_drug_details, but got {type(target_drug_details)}")
        return "Error: Invalid target drug details format."

    key = _cache_key(target_drug_details, similar_drugs_details)
    cached = _cache_get(key)
    if cached is not None:
        logging.info(f"Recommendation cache hit for target: {target_drug_details.get('name')}")
        return cached

    client, error = _get_client()
    if client is None:
        return f"Error: {error}"

    from google.genai import types

    prompt = _build_prompt(target_drug_details, similar_drugs_details)
    logging.info(f"Sending prompt to Gemini for target: {target_drug_details.get('name')}")

    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=400,
                # Low temperature: the same question should get the same answer.
                temperature=0.2,
                # 2.5-series models think by default, and thinking tokens count
                # against max_output_tokens -- leaving it on with a small budget
                # yields an empty response. This is a short extraction task.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        result_text = getattr(response, "text", None)
        if not result_text:
            reason = "empty response"
            candidates = getattr(response, "candidates", None)
            if candidates:
                finish_reason = getattr(candidates[0], "finish_reason", None)
                if finish_reason:
                    reason = f"finished due to: {finish_reason}"
            else:
                feedback = getattr(response, "prompt_feedback", None)
                block_reason = getattr(feedback, "block_reason", None) if feedback else None
                if block_reason:
                    reason = f"blocked due to: {block_reason}"
            logging.warning(
                f"Gemini returned no text for target {target_drug_details.get('name')} ({reason})."
            )
            return f"Could not generate recommendation. Reason: {reason}."

        logging.info(f"Received Gemini response for target: {target_drug_details.get('name')}")
        answer = result_text.strip()
        # Only successful answers are cached; errors should be retried.
        _cache_put(key, answer)
        return answer

    except Exception as e:
        logging.exception("Gemini API call failed with an exception.")
        return f"Error: Failed to get recommendation from LLM ({type(e).__name__})."
