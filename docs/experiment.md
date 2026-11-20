# Experiment: ordering FineWeb-Edu by its classifier score

## Question

Holding the corpus and the token budget fixed, does the order in which a small language model sees
its documents change what it learns? Four orders of the same pool are compared: shuffled,
ascending score, descending score, and a two-stage cooldown that ends on the highest-scoring
documents.

## Arms

All four arms train on the same pool of documents, for the same number of tokens, with the same
model, optimizer, schedule and seed. They differ only in the order of the documents.

| Arm | Order |
|---|---|
| `shuffle` | A random order of all documents |
| `ascending` | Documents sorted by score, lowest first |
| `descending` | Documents sorted by score, highest first |
| `cooldown` | The lowest-scoring documents that fill 80% of the tokens, shuffled; then the rest, shuffled |

The cooldown split is taken by tokens at a document boundary, so the first stage holds at most 80%
of the training tokens, short by less than one document. Its final stage is shuffled rather than
sorted, so the arm tests a two-stage cooldown rather than a gradual curriculum.

Each document's score is the continuous `score` column of
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`), read as
published. Nothing is re-scored. The pool holds the documents whose rounded score is at least 3, that
is `score >= 2.5`.

Documents are joined with the `<|endoftext|>` token and cut into 1,024-token sequences. Each arm uses
exactly 1,716 steps of 524,288 tokens, which is 899,678,208 tokens. Each arm's shard holds one more
token, so that the last input has its next-token target. Only the last document of an arm can be
cut, at that boundary.

## Model

GPT-2 with 6 layers, width 512, 8 heads and a 1,024-token block, with tied input and output
embeddings. It has about 45.2M parameters, 18.9M of them outside the embeddings. This is about 20
tokens per parameter.

## Training setup

| Setting | Value |
|---|---|
| Tokens per arm | 900M (1,716 steps of 524,288 tokens) |
| Micro-batch x sequence | 8 x 1024, with 64 accumulation steps (16 x 1024 runs out of memory on this GPU) |
| Optimizer | AdamW, weight decay 0.1 on matrices |
| Learning rate | Peak 6e-4, 18 warmup steps (about 1% of the run), cosine decay to 6e-5 |
| Precision | bf16 autocast, `torch.compile` on the training forward |
| Seed | 42 |

## Evaluation

Evaluated every 50 steps and after the last step, 36 points per arm.

- **Primary metric: score-stratified validation loss.** Two million held-out tokens, stratified by
  score: 500,000 tokens in each of four buckets, [2.5, 3.0), [3.0, 3.5), [3.5, 4.0) and [4.0, 5.0].
  They are drawn from the rows after the pool, until each bucket is full. Each bucket's loss is
  logged as `val_loss_25_30`, `val_loss_30_35`, `val_loss_35_40` and `val_loss_40_50`. The overall
  loss, `val_loss`, is the mean over all 2M tokens. Each bucket's token count is in
  `validation/buckets.json`.
- **Benchmarks:** HellaSwag (full validation set, 10,042 examples), PIQA (validation, 1,838) and
  ARC-Easy (test, 2,376), each scored by normalized accuracy. HellaSwag is scored as in the original
  repository: the ending with the lowest average per-token loss is the prediction.

## Checkpoints

Saved locally every 50 steps, and published to the model repository every 100 steps. An interrupted
run resumes from its last checkpoint on the same data, fetching it from the repository when it is
missing.

## Limitations

- One seed per arm. Differences smaller than seed-to-seed noise cannot be resolved.
- The pool and the validation buckets come from the same FineWeb-Edu stream. Together they draw about
  900M tokens through a 10,000-document shuffle buffer, roughly 1.3 of the sample's 14 files. They
  test generalization within that slice, not to other web text.
- Each arm's last document is cut at the token budget, so two arms can differ by part of one
  document.
- The score is a learned proxy for educational quality, so the buckets measure the classifier's
  judgement as much as quality.
