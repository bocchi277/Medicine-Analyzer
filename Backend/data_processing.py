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
import re

import pandas as pd

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Columns whose text feeds the vector for each drug. Changing these, or
# build_embedding_text(), invalidates index.faiss: rerun build_index.py.
EMBEDDING_TEXT_COLUMNS = [
    'name', 'drug_categories', 'mechanism', 'indication', 'pharmacodynamics',
    'background',
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
        RUNTIME_COLUMNS + (EMBEDDING_TEXT_COLUMNS + ['atc_codes'] if for_index else [])
    ))

    logging.info(f"Loading data from {config.DATA_PATH} (for_index={for_index})")
    read_names, rename_map, missing, header = _resolve_columns(requested)

    if 'drug_id' not in header:
        raise ValueError("Data must contain a 'drug_id' column.")

    try:
        # Everything is treated as text; dtype=str skips pandas' type inference
        # and avoids numeric columns coming back as floats like '1.0'.
        # The CSV writes missing values as the literal text "Not Available"
        # (in 98% of rows, often several times). Read it as missing, otherwise
        # that phrase ends up inside the vectors and makes drugs with sparse
        # records look alike. keep_default_na=False stops pandas from also
        # treating strings such as "NA" or "None" as missing.
        df = pd.read_csv(config.DATA_PATH, usecols=read_names, dtype=str,
                         keep_default_na=False, na_values=['', DISPLAY_PLACEHOLDER])
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
        df['combined_embedding_text'] = build_embedding_text(df, atc_group_names(df))

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


_TARGETS_HEADER = re.compile(r'\n?TARGET ACTIONS ORGANISM\n?')


def split_mechanism(text):
    """
    Split DrugBank's mechanism field into (prose, [target lines]).

    The field ends with a table flattened to lines, e.g.
        TARGET ACTIONS ORGANISM
        AProstaglandin G/H synthase 2 inhibitor Humans
    The first letter is a flag (A = known action, U = unknown), not part of
    the target name.
    """
    if not isinstance(text, str) or not text:
        return '', []
    parts = _TARGETS_HEADER.split(text, maxsplit=1)
    prose = parts[0].strip()
    targets = []
    if len(parts) > 1:
        for line in parts[1].split('\n'):
            line = line.strip()
            if len(line) > 1 and line[0] in 'AU':
                targets.append(line[1:].strip())
    return prose, targets


_ATC_NAME = re.compile(r'—\s*([^\n]+)')

# Categories about how a drug is metabolised or which side effects it shares
# ("Cytochrome P-450 CYP3A4 Substrates", "QTc Prolonging Agents"). They are
# common (CYP3A4 alone tags 866 drugs) and say nothing about what a drug is
# used for, but they filled the model's input. Paracetamol's text began with
# 20 of them, and its nearest neighbours were ketamine and estradiol.
_NON_THERAPEUTIC_CATEGORY = re.compile(
    r'^(Cytochrome P-450|P-glycoprotein|UGT|BCRP|OATP|OAT\d|OCT\d|MATE|BSEP|NTCP|'
    r'Drugs that are|Agents causing|Agents that produce|Agents that reduce|Agents producing|'
    r'Narrow Therapeutic Index|.*QTc|.*Associated Agents$|Myelosuppressive|Nephrotoxic|'
    r'Hepatotoxic|Neurotoxic|Ototoxic|Photosensitizing|Hypotensive Agents$|Hyperglycemia|'
    r'Hypoglycemia|Bradycardia|Serotonergic Drugs Shown|Drugs causing)', re.I)


def atc_group_names(df):
    """
    Every ATC group name in the data, lowercased ("cardiac therapy",
    "antiinflammatory and antirheumatic products, non-steroids", ...).

    DrugBank copies these names into drug_categories for 99% of drugs that
    have ATC codes. ATC codes are the answer key in evaluate.py, so leaving
    the names in the embedding text would let the model read the answer and
    inflate the score (36.8% -> 51.4% top-1 with the old model, most of it
    from this leak). They are removed so the evaluation stays independent.
    """
    names = set()
    if 'atc_codes' in df.columns:
        for value in df['atc_codes'].dropna():
            names.update(m.strip().lower() for m in _ATC_NAME.findall(value))
    return names


def build_embedding_text(df, exclude_categories=frozenset()):
    """
    One text per drug, with the most informative fields first.

    The old text started with name, generic name, approval groups and
    description, and the old model (paraphrase-MiniLM-L3-v2) read only the
    first 128 tokens, so for a third of drugs the mechanism and categories
    never reached it. Order now:

        name. Categories: ... Targets: ... Indication: ... mechanism prose ...

    Therapeutic categories and molecular targets (e.g. "Prostaglandin G/H
    synthase 2 inhibitor", shared by NSAIDs) say most about what a drug does,
    so they go first. Synonyms are left out: they are name variants and only
    pulled drugs towards others with similar spelling. Categories that are
    ATC group names (see atc_group_names) or about metabolism and side
    effects (_NON_THERAPEUTIC_CATEGORY) are dropped.
    """
    def col(name):
        return df[name].fillna('') if name in df.columns else pd.Series('', index=df.index)

    mechanism = col('mechanism').map(split_mechanism)
    prose = mechanism.map(lambda m: m[0])
    targets = mechanism.map(lambda m: '; '.join(m[1]))
    categories = col('drug_categories').map(
        lambda v: '; '.join(c.strip() for c in v.split('\n')
                            if c.strip() and c.strip().lower() not in exclude_categories
                            and not _NON_THERAPEUTIC_CATEGORY.search(c.strip())))

    def labelled(label, series):
        return series.map(lambda v: f'{label}: {v}.' if v else '')

    text = (col('name') + '. ' + labelled('Categories', categories) + ' '
            + labelled('Targets', targets) + ' ' + labelled('Indication', col('indication'))
            + ' ' + prose + ' ' + col('pharmacodynamics') + ' ' + col('background'))
    return text.str.replace(r'\s+', ' ', regex=True).str.strip()
