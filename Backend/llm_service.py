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


def format_drug_info_for_prompt(drug_details):
    """Formats drug details nicely for the LLM prompt."""
    if not isinstance(drug_details, dict):
        logging.warning(f"Expected dict for drug_details, but got {type(drug_details)}")
        return "Invalid drug details format."

    info = f"- Name: {drug_details.get('name', 'N/A')}\n"
    info += f"  Generic Name: {drug_details.get('generic_name', 'N/A')}\n"
    info += f"  Indication: {drug_details.get('indication', 'N/A')}\n"
    info += f"  Mechanism: {drug_details.get('mechanism', 'N/A')}\n"
    info += f"  Pharmacodynamics: {drug_details.get('pharmacodynamics', 'N/A')}\n"
    info += f"  Metabolism: {drug_details.get('metabolism', 'N/A')}\n"
    info += f"  Route: {drug_details.get('routes_of_administration', 'N/A')}\n"
    info += f"  Protein Binding: {drug_details.get('protein_binding', 'N/A')}\n"
    return info


def _build_prompt(target_drug_details, similar_drugs_details):
    prompt = (
        "**Objective:** Recommend the single best alternative drug from the list provided, "
        "considering potential efficacy, mechanism, and general profile based ONLY on the "
        "information given. If the information is not sufficient then simply return that no "
        "alternate is available. Assume the user needs an alternative to "
        f"'{target_drug_details.get('name', 'the target drug')}'.\n\n"
    )

    prompt += "**Target Drug:**\n"
    prompt += format_drug_info_for_prompt(target_drug_details) + "\n"

    prompt += "**Potential Alternatives:**\n"
    for i, drug in enumerate(similar_drugs_details):
        if isinstance(drug, dict):
            prompt += f"{i + 1}. {format_drug_info_for_prompt(drug)}\n"
        else:
            logging.warning(f"Skipping invalid item in similar_drugs_details at index {i}: {type(drug)}")
            prompt += f"{i + 1}. Invalid drug data format.\n"

    prompt += "**Task:**\n"
    prompt += "1. Analyze the provided details for the target drug and the potential alternatives.\n"
    prompt += "2. Identify the **single best alternative** from the list (1, 2, 3, etc.).\n"
    prompt += (
        "3. Provide a **brief reasoning** (1-2 sentences) for your choice, referencing the "
        "provided data (e.g., similar mechanism, different route, etc.). Focus only on the "
        "data presented.\n"
    )
    prompt += (
        "4. Format the output clearly, starting with 'Best Alternative:' followed by the drug "
        "name, then 'Reasoning:'. If the data is too limited, state that an alternative "
        "medicine is not available for the queried medicine.\n"
    )
    prompt += "5. Also give the available brand names even if they are not present in the data.\n\n"
    prompt += "**Output:**\n"
    return prompt


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
                temperature=0.5,
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
        return result_text.strip()

    except Exception as e:
        logging.exception("Gemini API call failed with an exception.")
        return f"Error: Failed to get recommendation from LLM ({type(e).__name__})."
