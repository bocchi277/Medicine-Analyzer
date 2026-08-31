"""
Loads and cleans the drug dataset.

Two modes, because the web server and the offline index builder need different
columns:

  for_index=False (default, runtime)  -> only the columns the API serves.
  for_index=True  (build_index.py)    -> additionally builds the concatenated
                                         'combined_embedding_text' column.

Reading only the required columns matters: the full 23-column CSV costs ~60 MB
of peak RSS to parse, and the runtime never looks at 'groups', 'background',
'drug_categories', 'type', 'formula', 'weight', 'atc_codes', 'cas_number' or
'pharmacogenomics'.
"""

import logging

import pandas as pd

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Columns whose text carries semantic meaning, concatenated to produce the
# vector for each drug. Changing this list invalidates index.faiss -- rerun
# build_index.py afterwards.
EMBEDDING_TEXT_COLUMNS = [
    'name', 'generic_name', 'groups', 'description', 'indication',
    'mechanism', 'pharmacodynamics', 'background', 'drug_categories',
    'synonyms',
]

# Columns retrieved for display, search and LLM context.
#
# 'groups', 'background' and 'drug_categories' are also embedding sources, but
# they carry genuinely useful display content (approval status, a prose
# background, and therapeutic categories are populated for ~10k of 17.4k rows)
# and the pages were rendering "Not Available" for them. They cost ~9.8 MB.
RUNTIME_COLUMNS = [
    'drug_id', 'name', 'generic_name', 'brand_name', 'indication', 'mechanism',
    'metabolism', 'half_life', 'pharmacodynamics', 'routes_of_administration',
    'protein_binding', 'description', 'synonyms', 'smiles',
    'groups', 'background', 'drug_categories',
]

# Canonical column name -> names it may actually appear under in the CSV.
# The dataset ships brand names as 'brand_names' (plural). Without this mapping
# the loader silently invented an empty 'brand_name' column, so brand-name
# search matched nothing and the field always rendered as "Not Available".
COLUMN_ALIASES = {
    'brand_name': ['brand_names'],
}

# Missing text in a column used for embedding becomes '' so it contributes
# nothing; missing text in a display-only column becomes a visible placeholder.
DISPLAY_PLACEHOLDER = "Not Available"


def _resolve_columns(requested):
    """
    Map requested canonical column names onto the CSV's actual header.

    Returns (read_names, rename_map, missing):
        read_names  -- names to hand to read_csv(usecols=...)
        rename_map  -- {csv_name: canonical_name} for aliased columns
        missing     -- canonical names with no source column at all
    """
    header = pd.read_csv(config.DATA_PATH, nrows=0).columns.tolist()

    read_names, rename_map, missing = [], {}, []
    for col in requested:
        if col in header:
            read_names.append(col)
            continue
        source = next((a for a in COLUMN_ALIASES.get(col, []) if a in header), None)
        if source:
            read_names.append(source)
            rename_map[source] = col
            logging.info(f"Column '{col}' resolved from '{source}' in the data file.")
        else:
            missing.append(col)

    if missing:
        logging.warning(
            f"Columns requested but absent from {config.DATA_PATH}: {missing}. "
            "They will be filled with defaults."
        )
    return read_names, rename_map, missing, header


def load_and_preprocess_data(for_index=False):
    """
    Load the drug data, normalise its text columns, and index it by drug_id.

    Args:
        for_index: when True, also build 'combined_embedding_text'. Only the
            offline index builder needs this; it is the largest single column
            (~23 MB) and the runtime does not use it.

    Returns:
        pd.DataFrame indexed by drug_id (also retained as a column).

    Raises:
        FileNotFoundError: the configured data file does not exist.
        ValueError: required columns are absent, or no rows survive cleaning.
    """
    requested = list(dict.fromkeys(
        RUNTIME_COLUMNS + (EMBEDDING_TEXT_COLUMNS if for_index else [])
    ))

    logging.info(f"Loading data from {config.DATA_PATH} (for_index={for_index})")
    read_names, rename_map, missing, header = _resolve_columns(requested)

    if 'drug_id' not in header:
        raise ValueError("Data must contain a 'drug_id' column.")

    try:
        # Everything is treated as text; dtype=str skips pandas' type inference
        # and avoids numeric columns coming back as floats like '1.0'.
        df = pd.read_csv(config.DATA_PATH, usecols=read_names, dtype=str)
    except FileNotFoundError:
        logging.error(f"Data file not found at {config.DATA_PATH}.")
        raise
    except Exception as e:
        logging.error(f"Failed to read CSV from {config.DATA_PATH}: {e}")
        raise

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    logging.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

    df['drug_id'] = df['drug_id'].astype(str)

    # Fill rules are keyed off EMBEDDING_TEXT_COLUMNS regardless of mode, so a
    # column never gets one value when building the index and another at
    # runtime. Columns feeding the embedding must not gain placeholder text, or
    # the phrase "Not Available" ends up inside the vectors.
    for col in df.columns:
        if col == 'drug_id':
            continue
        if col in EMBEDDING_TEXT_COLUMNS:
            df[col] = df[col].fillna("")
        elif col == 'smiles':
            # Consumed by structure renderers, not read by humans; empty string
            # is the right "absent" value.
            df[col] = df[col].fillna("")
        else:
            df[col] = df[col].fillna(DISPLAY_PLACEHOLDER)

    # Add any column the schema did not provide, so downstream code can rely on
    # its presence.
    for col in missing:
        if col in EMBEDDING_TEXT_COLUMNS or col == 'smiles':
            df[col] = ""
        else:
            df[col] = DISPLAY_PLACEHOLDER
        logging.debug(f"Added missing column '{col}' with default value.")

    if for_index:
        cols_for_text = [c for c in EMBEDDING_TEXT_COLUMNS if c in df.columns]
        if not cols_for_text:
            raise ValueError("Cannot build combined embedding text: no source columns present.")
        logging.info(f"Building combined embedding text from: {cols_for_text}")
        df['combined_embedding_text'] = df[cols_for_text].agg(' '.join, axis=1)

    # Drop columns that were only pulled in to build the embedding text, so the
    # returned frame is the same shape in both modes plus the text column.
    surplus = [c for c in df.columns
               if c not in RUNTIME_COLUMNS and c != 'combined_embedding_text']
    if surplus:
        df.drop(columns=surplus, inplace=True)

    before = df.shape[0]
    df.drop_duplicates(subset=['drug_id'], keep='first', inplace=True)
    if df.shape[0] < before:
        logging.warning(f"Removed {before - df.shape[0]} duplicate 'drug_id' entries.")

    # drop=False keeps drug_id available both as the index (for .loc lookups)
    # and as a column (so it appears in API responses).
    df.set_index('drug_id', inplace=True, drop=False)

    if df.empty:
        raise ValueError("No data available after processing.")

    logging.info(f"Processed data shape: {df.shape}")
    return df
