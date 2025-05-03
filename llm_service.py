import google.generativeai as genai
import logging
from config import config # Assuming config.py exists and has GEMINI_API_KEY

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_gemini():
    """
    Configures the Gemini API and initializes the GenerativeModel.
    This function is called once during application startup.
    """
    logging.info("Attempting to configure Gemini API...")
    try:
        # --- CRITICAL CHECK 1: Verify config.GEMINI_API_KEY ---
        # Ensure this key is correct and valid. An incorrect key *could*
        # potentially lead to resource not found errors depending on the API.
        if not hasattr(config, 'GEMINI_API_KEY') or not config.GEMINI_API_KEY:
             logging.error("GEMINI_API_KEY not found or is empty in config.py!")
             raise ValueError("GEMINI_API_KEY is not set.")

        genai.configure(api_key=config.GEMINI_API_KEY)
        logging.info("Gemini API configured.")

        # --- DEBUGGING STEP 2: Check available models ---
        # Keep this commented out unless you need to list models again.
        # logging.info("Listing available Gemini models:")
        # try:
        #     for m in genai.list_models():
        #         if 'generateContent' in m.supported_generation_methods:
        #             logging.info(f"  Available model: {m.name}")
        # except Exception as list_e:
        #      logging.error(f"Failed to list models: {list_e}")
        # logging.info("Finished listing models.")
        # --- END DEBUGGING STEP 2 ---

        # --- CRITICAL CHECK 3: Verify Model Name ---
        # Based on your logs, 'models/gemini-pro' was not found.
        # We are changing it to 'models/gemini-1.5-pro-latest' which was listed.
        # If you want to try a different model from your logs (e.g., 'models/gemini-1.5-flash-latest'),
        # update this line accordingly.
        model_name = 'models/gemini-1.5-pro-latest' # <--- CHANGED THIS LINE
        logging.info(f"Initializing Gemini model: {model_name}")
        model = genai.GenerativeModel(model_name)
        logging.info(f"Gemini model '{model_name}' initialized successfully.")
        return model

    except Exception as e:
        logging.error(f"Failed to configure or initialize Gemini API model: {e}", exc_info=True) # Log traceback
        # Re-raise the exception so the main application knows initialization failed
        raise

# Configure once on import when the module is loaded
# If configuration fails, gemini_model will be None, and subsequent calls will return an error message
try:
    gemini_model = configure_gemini()
except Exception:
    # The exception is already logged in configure_gemini
    gemini_model = None # Set to None so the API route can check


def format_drug_info_for_prompt(drug_details):
    """Formats drug details nicely for the LLM prompt."""
    # Ensure drug_details is a dictionary and handle potential missing keys gracefully
    if not isinstance(drug_details, dict):
        logging.warning(f"Expected dict for drug_details, but got {type(drug_details)}")
        return "Invalid drug details format."

    # Select and format relevant details from the full drug data dictionary
    # Use .get() with a default value to avoid KeyError if a key is missing
    info = f"- Name: {drug_details.get('name', 'N/A')}\n"
    info += f"  Generic Name: {drug_details.get('generic_name', 'N/A')}\n"
    info += f"  Indication: {drug_details.get('indication', 'N/A')}\n"
    info += f"  Mechanism: {drug_details.get('mechanism', 'N/A')}\n"
    # Add richer context - ensure these keys might exist based on your data processing
    info += f"  Pharmacodynamics: {drug_details.get('pharmacodynamics', 'N/A')}\n"
    info += f"  Metabolism: {drug_details.get('metabolism', 'N/A')}\n"
    info += f"  Route: {drug_details.get('routes_of_administration', 'N/A')}\n"
    info += f"  Protein Binding: {drug_details.get('protein_binding', 'N/A')}\n"
    # Consider adding brand_name and synonyms if you want the LLM to see them
    # info += f"  Brand Name: {drug_details.get('brand_name', 'N/A')}\n"
    # info += f"  Synonyms: {drug_details.get('synonyms', 'N/A')}\n"

    return info

def get_best_alternative_from_llm(target_drug_details, similar_drugs_details):
    """
    Queries the Gemini API to suggest the best alternative.

    Args:
        target_drug_details (dict): Dictionary with details of the original drug.
        similar_drugs_details (list[dict]): List of dictionaries with details of similar drugs.

    Returns:
        str: The LLM's recommendation text or an error message.
    """
    # Check if the model was successfully configured on startup
    if gemini_model is None:
        logging.error("Gemini model is None. Configuration failed on startup.")
        return "Error: LLM service not available due to configuration failure."

    # Check if there are any similar drugs to evaluate
    if not similar_drugs_details:
        logging.info("No similar drugs provided to LLM for evaluation.")
        return "No similar drugs found to evaluate."

    # --- Build the Prompt ---
    # Ensure target_drug_details is a dict before accessing it
    if not isinstance(target_drug_details, dict):
         logging.error(f"Expected dict for target_drug_details, but got {type(target_drug_details)}")
         return "Error: Invalid target drug details format."

    prompt = f"**Objective:** Recommend the single best alternative drug from the list provided, considering potential efficacy, mechanism, and general profile based ONLY on the information given.If the information is not sufficient then simply return that no alternate is available, Assume the user needs an alternative to '{target_drug_details.get('name', 'the target drug')}'.\n\n"

    prompt += "**Target Drug:**\n"
    prompt += format_drug_info_for_prompt(target_drug_details) + "\n"

    prompt += "**Potential Alternatives:**\n"
    if not similar_drugs_details:
         prompt += "No alternatives provided.\n" # Should be caught earlier, but defensive
    else:
        for i, drug in enumerate(similar_drugs_details):
            # Ensure each item in the list is a dict
            if isinstance(drug, dict):
                prompt += f"{i+1}. {format_drug_info_for_prompt(drug)}\n"
            else:
                logging.warning(f"Skipping invalid item in similar_drugs_details list at index {i}: {type(drug)}")
                prompt += f"{i+1}. Invalid drug data format.\n"


    prompt += "**Task:**\n"
    prompt += "1. Analyze the provided details for the target drug and the potential alternatives.\n"
    prompt += "2. Identify the **single best alternative** from the list (1, 2, 3, etc.).\n"
    prompt += "3. Provide a **brief reasoning** (1-2 sentences) for your choice, referencing the provided data (e.g., similar mechanism, different route, etc.). Focus only on the data presented.\n"
    prompt += "4. Format the output clearly, starting with 'Best Alternative:' followed by the drug name, and then 'Reasoning and if the data is too limited then simply return that the alternate medicine is not available for the query medicine:'.\n"
    prompt += "5. Also give the available brand names even if they are not available in the data.'.\n\n"
    prompt += "**Output:**\n"
    # --- End of Prompt ---


    logging.info(f"Sending prompt to Gemini for target: {target_drug_details.get('name')}")
    # Uncomment for detailed prompt debugging:
    # logging.debug(f"Gemini Prompt:\n{prompt}")

    try:
        # --- API Call ---
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                # candidate_count=1, # Default is 1
                # stop_sequences=['\n\n'], # Optional: Stop generation earlier
                max_output_tokens=250, # Adjust as needed
                temperature=0.5 # Lower temperature for more factual/focused response
            )
        )
        # --- End API Call ---

        # Handle potential safety blocks or lack of response text
        # Check response structure carefully based on google-generativeai library
        if not response or not response.candidates:
             logging.warning(f"Gemini response object is empty or has no candidates for target: {target_drug_details.get('name')}. Prompt feedback: {response.prompt_feedback}")
             return f"Could not generate recommendation. Reason: Empty response or no candidates."

        # Check if the first candidate and its content/parts are available
        if not hasattr(response.candidates[0], 'content') or not hasattr(response.candidates[0].content, 'parts') or not response.candidates[0].content.parts:
             logging.warning(f"Gemini response candidate content is missing or empty for target: {target_drug_details.get('name')}. Finish reason: {response.candidates[0].finish_reason}. Prompt feedback: {response.prompt_feedback}")
             safety_reason = "Response content empty."
             if hasattr(response.candidates[0], 'finish_reason') and response.candidates[0].finish_reason != 'STOP':
                  safety_reason = f"Finished due to: {response.candidates[0].finish_reason}"
             elif response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                  safety_reason = f"Blocked due to: {response.prompt_feedback.block_reason}"

             return f"Could not generate recommendation. Reason: {safety_reason}"


        # If we reach here, response and content should be available
        result_text = response.text
        logging.info(f"Received Gemini response for target: {target_drug_details.get('name')}")
        # Uncomment to see the raw LLM output:
        # logging.debug(f"Raw Gemini Response:\n{result_text}")
        return result_text.strip()

    except Exception as e:
        logging.error(f"Error querying Gemini API: {e}")
        # Log the type of error for better debugging
        logging.exception("Gemini API call failed with an exception.") # Logs the full traceback
        # Return a specific error message including the exception type
        return f"Error: Failed to get recommendation from LLM ({type(e).__name__})."

