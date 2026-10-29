"""
Measure search quality against ATC codes.

ATC (Anatomical Therapeutic Chemical) is the WHO's classification of drugs by
organ system and therapeutic use. The dataset has ATC codes for 3,541 drugs,
and they are never used to build the vectors, so they work as an independent
answer key: a good neighbour of a drug should share its ATC group.

Two numbers per ATC level:
  top1  - share of drugs whose nearest neighbour shares a code at that level
  top10 - share of the 10 nearest neighbours that do

Two neighbour pools:
  all   - neighbours from all 17,436 drugs, as the API searches. Neighbours
          without any ATC code always count as misses, so this is strict.
  atc   - neighbours only from drugs that have ATC codes.

    python evaluate.py                 # evaluate index.faiss
    python evaluate.py --json out.json # also save the numbers
"""
import argparse
import json
import pickle
import re

import faiss
import numpy as np
import pandas as pd

from config import config

LEVELS = {"L1 organ system": 1, "L2 therapeutic group": 3,
          "L3 pharmacological subgroup": 4, "L4 chemical subgroup": 5}
_CODE = re.compile(r"\b([A-Z]\d\d[A-Z]{0,2}\d{0,2})\s+—")


def load_atc(csv_path=None):
    df = pd.read_csv(csv_path or config.DATA_PATH, usecols=["drug_id", "atc_codes"],
                     dtype=str, keep_default_na=False)
    return {row.drug_id: set(_CODE.findall(row.atc_codes)) for row in df.itertuples()}


def atc_scores(index, id_map, atc_by_drug, pool="all", k=10):
    positions = sorted(id_map)
    vectors = np.vstack([index.reconstruct(p) for p in positions]).astype("float32")
    faiss.normalize_L2(vectors)
    drug_ids = [id_map[p] for p in positions]
    codes = [atc_by_drug.get(d, set()) for d in drug_ids]
    queries = [i for i, c in enumerate(codes) if c]

    if pool == "atc":
        pool_rows = queries
    else:
        pool_rows = list(range(len(drug_ids)))
    search = faiss.IndexFlatIP(vectors.shape[1])
    search.add(vectors[pool_rows])
    _, found = search.search(vectors[queries], k + 1)

    results = {}
    for label, length in LEVELS.items():
        top1, topk = [], []
        for qi, row in zip(queries, found):
            mine = {c[:length] for c in codes[qi] if len(c) >= length}
            neighbours = [pool_rows[j] for j in row if j >= 0 and pool_rows[j] != qi][:k]
            same = [bool(mine & {c[:length] for c in codes[n] if len(c) >= length})
                    for n in neighbours]
            top1.append(same[0])
            topk.append(sum(same) / len(same))
        results[label] = {"top1": round(100 * float(np.mean(top1)), 1),
                          f"top{k}": round(100 * float(np.mean(topk)), 1)}
    return {"drugs_evaluated": len(queries), "pool": pool, "levels": results}


def load_index():
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.ID_MAP_PATH, "rb") as f:
        id_map = pickle.load(f)
    return index, id_map


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", help="write results to this file")
    args = parser.parse_args()

    index, id_map = load_index()
    atc = load_atc()
    report = {pool: atc_scores(index, id_map, atc, pool=pool) for pool in ("all", "atc")}
    for pool, result in report.items():
        print(f"\nNeighbour pool: {pool} ({result['drugs_evaluated']} drugs with ATC codes)")
        for level, scores in result["levels"].items():
            print(f"  {level:30s} top1 {scores['top1']:5.1f}%   top10 {scores['top10']:5.1f}%")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
