# Hub census

What do the weights on the Hugging Face Hub actually say about themselves, and how often does the
config agree? This folder reads the safetensors headers of a **stratified sample** of Hub repos
(a few hundred KB per repo, no weights downloaded) and weights the results back to the whole Hub.

| step | file | output |
|---|---|---|
| 1. frame | `frame.py` | every public model with downloads (30-day and all-time), likes, created/modified dates, library, task, tags, gated flag. From the daily [hub-stats](https://huggingface.co/datasets/cfahlgren1/hub-stats) dump, or the Hub API with `--api` |
| 2. sample | `sample.py` | the top N by downloads (read in full) + a random sample of the rest, stratified by year × download tier × safetensors tag, with weights |
| 3. crawl | `crawl.py` | one JSONL line per repo: status, family from weights alone and from config, params vs the Hub's own count, quantization, MoE / SSM, config disagreements, bytes read, time |
| 4. summarize | `summarize.py` | `out/report/summary.md` with 95% intervals, `status.png`, `cost.png` |

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

## Run

On GitHub: **Actions → Hub census (pilot) → Run workflow**. Results are attached to the run as an
artifact, and the summary appears on the run page. Add a repository secret `HF_TOKEN` (read-only) for
higher rate limits.

Locally:

```bash
pip install ai-model-explorer pandas pyarrow
python frame.py && python sample.py --top 500 --rest 500 && python crawl.py && python summarize.py
```
