# Customer Voice Intelligence

**A Deep Learning Framework for Review Sentiment Analysis, Cross-Domain Transfer, and Foundation Model Architecture in Streaming and E-Commerce**

Amol Birajdar · Walsh College · QM 640 Data Analytics Capstone · Mentor: Dr. Rishabh Pandey

## Notebooks (run in this order in Google Colab)

| Notebook | Purpose | Runtime | Report outputs |
|---|---|---|---|
| `notebooks/01_IMDb_Data_Exploration.ipynb` | Download IMDb v1.0, clean, split, EDA | CPU | Tables 3–4, Figures 1–5, 9–10 |
| `notebooks/02_Amazon_Sampling.ipynb` | Stream Amazon Reviews 2023 Electronics, 20k balanced sample, RAG products | CPU | Figures A1–A2 |
| `notebooks/03_RQ1_Architecture_Comparison.ipynb` | TF-IDF LogReg, FFN, 1D-CNN, BiLSTM; 5 seeds; McNemar; latency; score masking | GPU | Tables 7–9, Figures 6–8 |
| `notebooks/04_RQ2_Transfer_Learning.ipynb` | DistilBERT fine-tuning, domain shift, learning curves | GPU | Tables 10–11, Figures 11–12 |
| `notebooks/05_RQ3_Anomaly_Detection.ipynb` | Autoencoder / VAE quality gate, injected anomalies, VAE augmentation | GPU | Tables 12–15, Figures 13–14 |
| `notebooks/06_RQ4_Foundation_Models.ipynb` | Zero/few-shot LLM, RAG summaries, break-even | GPU | Tables 16–18, Figures 15–16 |
| `notebooks/07_Final_Summary.ipynb` | Research-question answers, tiered architecture, cascade | CPU | Tables 19–20, Figures 17–18 |

All notebooks mount Google Drive and share one folder, `MyDrive/customer-voice-intelligence/`:

```
customer-voice-intelligence/
├── code/            cvi_utils.py (written by 01), cvi_models.py (written by 03)
├── data/raw/        aclImdb_v1.tar.gz
├── data/processed/  imdb_clean.csv, amazon_electronics_20k.csv, amazon_rag_products.csv, rq3_eval_set.csv
├── models/          rq1/, rq2/, rq3/
└── outputs/         figures/, tables/, results/
```

Long experiments save each run as it finishes. If Colab disconnects, re-run the notebook and it skips the runs that are already done.

## Data sources
* IMDb Large Movie Review Dataset v1.0 (Maas et al., 2011): https://ai.stanford.edu/~amaas/data/sentiment/
* Amazon Reviews 2023 (Hou et al., 2024): https://amazon-reviews-2023.github.io/ (files via https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

No Kaggle or scraped data are used. The Amazon `user_id` field is dropped on read.

## Settings to review before the final run
* `03`: `COSTS` (cloud CPU/GPU hourly prices used for cost per 1,000 reviews).
* `02`: `MAX_SCAN` (how much of the Electronics file is scanned; `None` = all).
* `06`: `PROVIDER` (`hf_local` is free; `anthropic` or `openai` need an API key in Colab Secrets) and `API_PRICES`; `LABEL_COST_USD`.
