# Experiment: soft weighting versus hard filtering

## Question

FineWeb-Edu keeps documents whose classifier score is 3 or higher and discards the rest. This is a hard filter: a document either counts fully or not at all. Does a soft alternative do better? Here, every document counts in proportion to its score.

## Arms

All three arms train the same GPT-2 (124M) for the same number of tokens, with the same seed, optimizer and schedule. They differ only in which documents they see and how much each token counts.

| Arm | Documents | Loss weight per token |
|---|---|---|
| `uniform` | All FineWeb documents | 1 |
| `hard` | FineWeb-Edu documents (score 3 to 5) | 1 |
| `soft` | The same FineWeb documents as `uniform` | score / 5 |

`uniform` and `soft` see identical text in identical order, so their difference is the weighting alone. `hard` compares filtering with the other two.

Each document's score comes from the [FineWeb-Edu classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier), rounded to an integer from 0 to 5, applied to the first 512 tokens of the document.

## Training setup

| Setting | Value |
|---|---|
| Tokens per arm | 100M (190 optimizer steps of 524,288 tokens) |
| Micro-batch x sequence | 16 x 1024, with 32 accumulation steps |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 on matrices |
| Learning rate | Peak 6e-4, 10 warmup steps, cosine decay to 6e-5 |
| Precision | bf16 autocast, `torch.compile` on the training forward |
| Seed | 42 |
| Validation | 40 batches of 16 from held-out FineWeb documents, unweighted, every 10 steps |
| Benchmark | Full HellaSwag validation set (10,042 examples), normalized accuracy |

## Loss normalization

Within each micro-batch the weighted loss is the sum of `weight x loss` divided by the sum of weights. With unit weights this is the ordinary mean, so `uniform` and `hard` are unaffected. A micro-batch whose weights all vanish gives zero loss, not NaN.

## Limitations

- One seed per arm. Differences smaller than seed-to-seed noise cannot be resolved.
- 100M tokens is far below the 124M model's capacity, and HellaSwag is near chance at this scale. Expect small effects.
- The `hard` arm uses the published FineWeb-Edu documents, which are not the same set of documents as the `raw` arm. Any difference between `hard` and the others mixes filtering with document choice.
- The classifier is itself a learned proxy for educational quality.

## Results

Results are published with the runs once training finishes; see the [model repository](https://huggingface.co/NoeFlandre/GPT2-Fineweb-Edu-Experiments).

## Data balance checks

Measured on the shards used for the runs (per-document scores, from the classifier for `raw`):

| Shard | Documents | Mean length (tokens) | Documents by score 0 / 1 / 2 / 3 / 4 / 5 |
|---|---|---|---|
| `raw/train` (uniform, soft) | 143,468 | 697 | 12% / 62% / 20% / 5% / 0.8% / 0% |
| `raw/val` (validation) | 3,008 | 665 | 12% / 62% / 20% / 5% / 0.5% / 0% |
| `edu/train` (hard) | 95,821 | 1,044 | 0% / 0% / 0% / 87% / 13% / 0.1% |

The raw training and validation shards have matching score mixes. The `hard` arm uses a different document source and longer documents, so its comparison with `uniform` also reflects that difference. This is accepted for the current run.
