# GPT-2 from scratch on FineWeb-Edu

A GPT-2 (124M) implementation trained on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), built to run experiments on a single laptop GPU.

The current experiment asks one question: **does weighting each document by its educational score help more than filtering out low-scoring documents?** See [the experiment](experiment.md).

## Quick start

```bash
uv sync
uv run pytest            # unit and integration tests
uv run ruff check        # lint
uv run ty check          # types
bash scripts/run_experiment.sh   # build data, train the three arms, plot
```

## Where things live

| Path | Contents |
|---|---|
| `gpt2_scratch/` | Library code: model, data, experiment, training, reporting |
| `scripts/` | Entry points: data preparation, training, reporting, upload |
| `tests/` | Test suite (TDD; kept at 100% line and branch coverage) |
| `docs/` | This documentation |
| `data/hellaswag/` | HellaSwag validation examples |

Results and checkpoints are published at [NoeFlandre/GPT2-Fineweb-Edu-Experiments](https://huggingface.co/NoeFlandre/GPT2-Fineweb-Edu-Experiments).
