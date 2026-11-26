# Results

File names are `YYYYMMDD-HHMM-<what>-<tag>`. Each answer run is a `.jsonl` file (one answer per line, with its passages, citations, flags and timings) plus a `.summary.json`.

## Sealed evaluations

Each question set was written from the corpus and committed before the system was run on it. Then the answers from every run were shuffled together with the run names hidden, and two LLM graders checked each one against the corpus using [`eval/RUBRIC.md`](../eval/RUBRIC.md). An answer is *fully right* only if both graders say yes, and *main point right* only if neither says no.

```bash
python eval/score_sealed.py heldout5 gpu8 gpu16
```

| Set | Code | Run | Main point | Fully right | Files |
|---|---|---|---:|---:|---|
| `heldout5` (98 Q) | 1f898d0 | gemma4:12b | 91.8% | 82.7% | `20261126-2100-answers-sealed-gpu16-heldout5*` |
| | | qwen3.5:4b | 89.8% | 72.4% | `20261126-2115-answers-sealed-gpu8-heldout5*` |
| `heldout4` (58 Q) | d8d1d84 | gemma4:12b | 98.3% | 93.1% | `20261126-1832-answers-sealed-gpu16-heldout4*` |
| | | qwen3.5:4b | 96.6% | 84.5% | `20261126-1843-answers-sealed-gpu8-heldout4*` |
| `heldout3` (99 Q) | a6a2c02 | gemma4:12b | 96.0% | 92.9% | `20261126-1440-answers-final-gpu16-heldout3*` |
| | | qwen3.5:4b | 90.9% | 81.8% | `20261126-1443-answers-final-gpu8-heldout3*` |
| | earlier code | qwen3.5:4b | 91.9% | 78.8% | `20261125-1744-answers-baseline-heldout3*` |

The blind batches, both graders' verdicts (each with a note and an evidence quote), the run mapping and the scores are in `sealed/`.

The heldout4 and heldout5 runs shared the GPU with another workload, so the reranker ran on the CPU (fp32 instead of fp16). That puts total latency in those files at ~10 s; generation p50 was 1.4 s (16 GB) and 0.7–0.8 s (8 GB).

heldout5 was written to avoid the angles of the earlier sets, and has more multi-part, table and cross-document questions; it scores lower than heldout4.

## Other runs

Everything else in this folder is development runs on `golden.yaml`, `heldout.yaml` and `heldout2.yaml`, kept as they were. Those sets were used while building the system, so their scores are in-sample.
