# Hub census

What do the weights on the Hugging Face Hub actually say about themselves, and how often does the
config agree? This folder reads the safetensors headers of a **stratified sample** of Hub repos
(a few hundred KB per repo, no weights downloaded) and weights the results back to the whole Hub.

| step | file | output |
|---|---|---|
| 1. frame | `frame.py` | every public model with downloads (30-day and all-time), likes, created/modified dates, library, task, tags, gated flag. From the daily [hub-stats](https://huggingface.co/datasets/cfahlgren1/hub-stats) dump, or the Hub API with `--api` |
| 2. sample | `sample.py` | the top N by downloads and the top N **original releases** (the Hub's `base_model:` tags mark quantized, fine-tuned, merged and adapter copies), both read in full, + a random sample of the rest, stratified by year × download tier × safetensors tag, with weights |
| 3. crawl | `crawl.py` | one JSONL line per repo: status, family from weights alone and from config, params vs the Hub's own count, quantization, MoE / SSM, config disagreements, bytes read, time |
| 4. summarize | `summarize.py` | `out/report/summary.md` with 95% intervals, `status.png`, `cost.png` |

## Re-scoring without crawling again

`crawl.py` also writes `out/census_manifests.jsonl.gz`: every readable repo's tensor names, dtypes, shapes and config.
After a change to the recovery rules, `python rescore.py && python summarize.py` recomputes every weights-derived field
offline in minutes; popularity, status and timing fields are kept.

## Frontier stratum and the holdout question

`--frontier N --since DATE` adds the N most-downloaded original releases from leading providers created after DATE
(list in `sample.py`). It measures how the method holds up on the newest architectures. It is **not** a clean holdout
for releases before 2026-11-03: the recovery rules were last tuned on that date with census data in view. Rules are frozen
at commit `6aeadcd` (2026-11-03, "rules-2026-11-03"; `src/infer.js` and `src/loaders.js` at that commit); re-running with `--since 2026-11-04` later gives a prospective holdout on models the rules
have never seen.

## Publishing

**Actions → Publish datasets → Run workflow** with the run ID of a finished census run. It re-scores the run with the
current rules, scores per-tensor roles against the corpus (`roles.js`), publishes `<you>/hub-weight-census` and
`<you>/manifest-anatomy-corpus` (the labelled 510-architecture corpus, `research/data/corpus-510.jsonl.gz`), groups both
with the explorer as a static Space (`<you>/ai-model-explorer`) in a collection "AI Model Explorer" on your Hugging
Face profile, and refreshes the website's census page. Both datasets carry a flat parquet table, so the Hub's dataset
viewer can browse, search and filter them. It needs a
repository secret `HF_WRITE_TOKEN` with write access; the owner is read from the token.

### By hand


`python publish.py owner/name` uploads the run with a dataset card and datasheet (needs a write token). In the workflow,
fill the `publish` input and add a repository secret `HF_WRITE_TOKEN`.

## Why a sample and not every repo

A proportion's 95% margin is `1.96 · sqrt(p(1-p)/n)`: about ±2% at n≈2,400 and ±1% at n≈9,600,
whatever the size of the Hub. Most repos have almost no downloads, so the top few thousand cover most
real usage; that stratum is read in full. Run the pilot (500 + 500) first, then size the full run from
its failure and timing rates.

## Failures are data

Gated (401/403), deleted (404), no readable weights (GGUF/ONNX only), timeouts, and errors are all
recorded with the repo's popularity, so the report shows *what share of the Hub can be read this way*.
Rate limits (429) are retried with `Retry-After` and back-off. Each repo runs in its own process with a
hard time limit. The crawl appends as it goes, so it can be stopped and restarted.

## Ecosystem findings and their statistics

`ecosystem.py` (run by `insights.py`) measures the ecosystem from the weights: architecture adoption by release year,
KV cache per token per billion parameters, how sparse mixtures of experts became, how width and depth scale with size,
tied embeddings by size, storage precision, the share of repos and of downloads by relation to a parent, whether
derivatives keep the parent's architecture, how concentrated lineage is, metadata reliability, and the smallest device
each original language model fits. Shares carry Wilson 95% intervals, medians bootstrap intervals; trends over release
year use the Cochran-Armitage test and a logistic odds ratio per year, monotone trends Spearman's rho, shape scaling
log-log slopes, two groups Mann-Whitney. Output: `out/insights/ecosystem.json` and `eco_*.png` / `eco_*.pdf`.
Language models are decoder or prefix-LM anatomies above 50M parameters with a text-generating Hub pipeline (or none).
The Hub dates every repository created before March 2022 to that month, so the first year bin reads "up to 2022".

## Labels and the board

`labels.py` tags each readable repo from its Hub tags: leading provider or not, its relation to its parent (original,
fine-tune, quantized copy, merge, adapter), and whether it is built for Thai (TH, TH/EN, or multilingual with Thai).
Two corrections to the Hub's `base_model` relation: a derivative of the same owner's model (an instruct release of its own
base) counts as that owner's original, and a repo tagged quantized whose safetensors are not quantized (it carries an ONNX
export, say) counts as a fine-tune. `board_seed.py` fills the Deploy-Ready board from a census run and from
`board_extra.txt`, scoring every row with `submit.score`.

## Run

On GitHub: **Actions → Hub census (pilot) → Run workflow**. Results are attached to the run as an
artifact, and the summary appears on the run page. Add a repository secret `HF_TOKEN` (read-only) for
higher rate limits.

Locally:

```bash
pip install ai-model-explorer pandas pyarrow
python frame.py && python sample.py --top 500 --rest 500 && python crawl.py && python summarize.py
```
