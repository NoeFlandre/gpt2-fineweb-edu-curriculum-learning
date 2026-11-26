# Curriculum learning on FineWeb-Edu

A GPT-2 implementation, trained from scratch on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), built to run experiments on a single laptop GPU.

The current experiment asks: **with the corpus and the token budget fixed, does the order in which documents are seen, by their educational score, change what the model learns?** See [the experiment](experiment.md).

## Quick start

```bash
uv sync
uv run pytest                    # unit and integration tests
uv run ruff check                # lint
uv run ty check                  # types
bash scripts/run_experiment.sh   # build data, train the four arms, plot, upload
```

## Where things live

| Path | Contents |
|---|---|
| `gpt2_scratch/` | Library code: model, data, experiment, evaluation, training, reporting |
| `scripts/` | Entry points: data preparation, benchmarks, training, reporting, upload |
| `tests/` | Test suite (TDD; kept at 100% line and branch coverage) |
| `docs/` | This documentation |
| `data/benchmarks/` | HellaSwag, PIQA and ARC-Easy examples in JSON Lines |

Results and checkpoints are published at [NoeFlandre/gpt2-fineweb-edu-curriculum-learning](https://huggingface.co/NoeFlandre/gpt2-fineweb-edu-curriculum-learning).
