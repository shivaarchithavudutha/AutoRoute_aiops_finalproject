# Data Card
- **Source:** CFPB Consumer Complaint Database, narratives archive (public, US government). URL and exact export files used: ______
- **Collection:** `scripts/download_data.sh` (version-controlled); files tracked with DVC (`data/raw/train.dvc`).
- **Format:** CSV; columns used: Date received, Product, Consumer complaint narrative.
- **Size:** raw rows ____, rows with narrative ____, rows after cleaning ____, training/test rows ____ / ____ (see `data/interim/data_quality.json`).
- **Labels:** `Product`, merged into ___ classes using `label_map.yaml` (list the mapping and why).
- **Potential biases and limitations:** only complaints whose authors agreed to publish a narrative; heavy class imbalance (credit reporting is about half);
  product names changed over time; English only; US financial products; redaction tokens (XXXX) remove details.
- **Quality checks:** list the checks in `validate_data.py` and the values you obtained.
- **EDA findings:** paste `docs/eda/eda_summary.md` and the three charts (run `python src/eda.py`).
- **Split:** stratified 80/20, classes capped at `max_rows_per_class`; newer exports reserved for the drift demo.
