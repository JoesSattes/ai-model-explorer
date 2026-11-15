# Vocabulary tax

Tables behind the census page's *Vocabulary tax* section, published as the Hugging Face dataset `vocabulary-tax` by the
*Publish vocabulary tax and Space* workflow (`publish.py` writes the datasheet).

- `data/models.parquet`: one row per model (2,061), tokenizer family, token-table and total bytes, tax for Thai + English
- `data/model_language.parquet`: model x language x rule (script, or your own text at 1k-15k lines)
- `data/tokenizers.parquet`: tokenizer family x language x rule: rows removed and real re-split rate
- `data/quality.parquet`: six models x 11 languages, bits per character, greedy continuations and Belebele before and after trimming
- `data/certificate.parquet`, `data/merge_closure.parquet`: the trim certificate and merge closure against the real re-split rate

Compute the same numbers for any model, with its own tokenizer and your own text: `pip install ai-model-explorer`, then
`ai-model-explorer tax REPO --lang th --corpus my_text.txt`.
