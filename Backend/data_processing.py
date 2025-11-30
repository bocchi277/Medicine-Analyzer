import pandas as pd
import logging
from config import config # Assuming config.py exists and has DATA_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data():
    """
    Loads and preprocesses the drug data from the CSV file.

    Ensures relevant text columns are treated as strings and handles potential
    missing values for text combination and display. Avoids introducing
    "Not Available" into the combined embedding text. Includes 'brand_name'
    and 'synonyms' for potential search/display.
    Removes the specific cleaning logic for the 'mechanism' column that replaced "TARGET".

    Returns:
        pd.DataFrame: The processed DataFrame ready for embedding and retrieval.

    Raises:
        FileNotFoundError: If the data file specified in config.DATA_PATH is not found.
        Exception: For other errors during file loading or processing.
        ValueError: If the processed DataFrame is empty or lacks critical columns.
    """
    logging.info(f"Attempting to load data from {config.DATA_PATH}") # <-- Check this DATA_PATH in config.py
    try:
        df = pd.read_csv(config.DATA_PATH)
        logging.info("Data loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Data file not found at {config.DATA_PATH}. Please ensure the file exists.")
        raise # Re-raise the exception so the calling code knows loading failed
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading CSV from {config.DATA_PATH}: {e}")
        raise # Re-raise the exception

    logging.info(f"Initial data shape: {df.shape}")
    logging.info(f"Initial columns: {df.columns.tolist()}") # <-- Check if 'smiles' is in this list


    # Define columns potentially useful for semantic meaning (for embedding)
    # Includes synonyms as they contribute to understanding the drug
    text_cols_for_embedding = [
        'name', 'generic_name', 'groups', 'description', 'indication',
        'mechanism', 'pharmacodynamics', 'background', 'drug_categories',
        'synonyms'
    ]
    # Define columns to keep for retrieval and LLM context (for display/details)
    # Includes brand_name as it's used for searching and display
    # --- ENSURE 'smiles' IS IN THIS LIST IF YOU WANT IT IN THE FINAL DATAFRAME ---
    cols_to_keep = [
        'drug_id', 'name', 'generic_name', 'brand_name', 'indication', 'mechanism',
        'metabolism', 'half_life', 'pharmacodynamics', 'routes_of_administration',
        'protein_binding', 'description', 'synonyms', 'smiles' # <-- Make sure 'smiles' is here
    ]

    # --- Data Cleaning and Type Conversion ---
    logging.info("Starting data cleaning and type conversion...")

    # Ensure 'drug_id' is present and handle potential non-numeric IDs if necessary
    if 'drug_id' not in df.columns:
         logging.error("Required column 'drug_id' not found in the data.")
         raise ValueError("Data must contain a 'drug_id' column.")
    # Attempt to convert drug_id to string to be safe
    df['drug_id'] = df['drug_id'].astype(str)


    # Process columns for embedding first: fill NaNs with empty strings and ensure string type
    logging.info("Processing columns for embedding...")
    existing_text_cols_for_embedding = [col for col in text_cols_for_embedding if col in df.columns]
    for col in existing_text_cols_for_embedding:
         # Use .fillna('') before .astype(str) to ensure NaNs become empty strings
         df[col] = df[col].fillna("").astype(str)
         logging.debug(f"Processed '{col}' for embedding: filled NaNs and converted to string.")

    # Process columns to keep (that are not already processed for embedding):
    # Fill NaNs with "Not Available" for display purposes and ensure string type
    logging.info("Processing columns to keep for display...")
    # Use set difference to get columns in cols_to_keep that are NOT in text_cols_for_embedding
    cols_to_keep_only_display = [
        col for col in set(cols_to_keep) - set(existing_text_cols_for_embedding)
        if col in df.columns and col != 'drug_id' and col != 'smiles' # Exclude smiles here, handle separately
    ]
    for col in cols_to_keep_only_display:
         df[col] = df[col].fillna("Not Available").astype(str)
         logging.debug(f"Processed '{col}' for display: filled NaNs with 'Not Available' and converted to string.")

    # --- Specific handling for smiles ---
    # Ensure smiles column exists and is string type, fill NaNs with empty string if needed for consistency
    if 'smiles' in df.columns:
        if pd.api.types.is_string_dtype(df['smiles']):
             df['smiles'] = df['smiles'].fillna("") # Fill NaN smiles with empty string
             logging.debug("Processed 'smiles' column: filled NaNs with empty string.")
        else:
             logging.warning(f"Column 'smiles' exists but is not string dtype ({df['smiles'].dtype}), converting to string and filling NaNs.")
             df['smiles'] = df['smiles'].astype(str).fillna("")
    else:
         logging.warning("Column 'smiles' not found in the loaded data. Adding with empty strings.")
         df['smiles'] = "" # Add the column if missing


    # Handle any other columns in text_cols_for_embedding or cols_to_keep that were missing initially
    # This loop is less critical if the above specific handling is robust, but kept for safety
    all_relevant_cols = list(set(text_cols_for_embedding + cols_to_keep))
    for col in all_relevant_cols:
        if col not in df.columns:
            logging.warning(f"Column '{col}' not found in the loaded data. Adding with default values.")
            # Add missing columns with default values appropriate for their intended use
            if col in text_cols_for_embedding:
                 df[col] = "" # Default for embedding text
            elif col in cols_to_keep:
                 df[col] = "Not Available" # Default for display text


    # --- Removed the specific cleaning logic for 'mechanism' that replaced "TARGET" ---
    # The general NaN handling above ensures 'mechanism' is a string with NaNs as ""
    # If you need other specific cleaning for 'mechanism', add it here,
    # but avoid replacing content with "" if you want the original text for embedding.
    # For example, to just remove the word "TARGET" without replacing the whole string:
    # if 'mechanism' in df.columns:
    #     df['mechanism'] = df['mechanism'].str.replace("TARGET", "", regex=False).str.strip()
    #     logging.info("Applied specific cleaning to 'mechanism' column: removed 'TARGET'.")
    # --- End Removed Block ---


    # Create the combined text for embedding
    # Use the list of columns confirmed to exist and processed for embedding
    if not existing_text_cols_for_embedding:
         logging.error("No valid columns found for creating combined embedding text after processing.")
         # This should ideally not happen if data loading was successful and columns were added
         raise ValueError("Cannot create combined text without specified columns.")

    logging.info(f"Creating combined embedding text from columns: {existing_text_cols_for_embedding}")
    # Use .agg(' '.join) which automatically handles empty strings correctly
    df['combined_embedding_text'] = df[existing_text_cols_for_embedding].agg(' '.join, axis=1)
    logging.info("Combined embedding text created.")

    # --- Final DataFrame Preparation ---
    # Select and return relevant data
    # Ensure all cols_to_keep and necessary embedding cols ('drug_id', 'combined_embedding_text') are included
    final_cols = list(set(['drug_id', 'combined_embedding_text'] + cols_to_keep))
    # Filter final_cols to only include columns that exist in the DataFrame
    final_cols_existing = [col for col in final_cols if col in df.columns]

    # Ensure 'combined_embedding_text' is always included if it was created
    if 'combined_embedding_text' not in final_cols_existing and 'combined_embedding_text' in df.columns:
         final_cols_existing.append('combined_embedding_text')

    df_processed = df[final_cols_existing].copy()

    # Ensure unique drug_id - handle duplicates if necessary
    initial_rows = df_processed.shape[0]
    # Use .loc for index-based drop_duplicates if index is already set to drug_id
    # If not, use subset=['drug_id']
    if 'drug_id' in df_processed.columns:
        df_processed.drop_duplicates(subset=['drug_id'], keep='first', inplace=True)
        if df_processed.shape[0] < initial_rows:
            logging.warning(f"Removed {initial_rows - df_processed.shape[0]} duplicate 'drug_id' entries.")
    else:
        logging.warning("'drug_id' column not found for dropping duplicates.")


    # Set index for easy lookup by drug_id in the API
    # Ensure drug_id column exists before setting index
    if 'drug_id' in df_processed.columns:
        try:
            df_processed.set_index('drug_id', inplace=True, drop=False) # Keep drug_id as a column too
            logging.info("Set 'drug_id' as DataFrame index in processed data.")
        except Exception as e:
            logging.error(f"Error setting 'drug_id' as index in processed data: {e}")
    else:
        logging.warning("'drug_id' column not found in processed data for setting index.")


    logging.info(f"Processed data shape: {df_processed.shape}")
    if df_processed.empty:
        logging.error("Processed DataFrame is empty after cleaning. Check input data and filters.")
        raise ValueError("No data available after processing.")

    # --- DEBUG LOGGING: Final check for smiles column in returned data ---
    if 'smiles' in df_processed.columns:
        logging.info("Data processing complete: 'smiles' column FOUND in returned DataFrame.")
    else:
        logging.error("Data processing complete: 'smiles' column NOT FOUND in returned DataFrame.")
    # --- END DEBUG LOGGING ---


    logging.info("Data preprocessing complete.")
    return df_processed

# Example of how you might call this function in your main app file:
# try:
#     drug_data_df = load_and_preprocess_data()
# except Exception as e:
#     logging.critical(f"Failed to load and preprocess data: {e}")
#     # Handle the error appropriately, maybe exit or disable API functionality
#     drug_data_df = pd.DataFrame() # Ensure drug_data_df is defined even on error
#     # Depending on severity, you might want to sys.exit(1) here
