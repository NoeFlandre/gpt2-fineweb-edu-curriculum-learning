import json

import torch
from gpt2_scratch.evaluation import hellaswag


def encode_bytes(text):
    return list(text.encode())


def example(label):
    return {"ctx": "ab", "endings": ["cd", "efg", "h", "ij"], "label": label}


def test_mask_covers_only_the_ending_tokens():
    tokens, mask, label = hellaswag.render_example(example(1), encode_bytes)

    assert tokens.shape[0] == 4
    assert mask[0].sum().item() == len(" cd")
    assert mask[1].sum().item() == len(" efg")
    assert mask[:, :2].sum().item() == 0
    assert label == 1


def test_context_is_shared_by_all_four_rows():
    tokens, _, _ = hellaswag.render_example(example(0), encode_bytes)

    assert torch.equal(tokens[:, :2], torch.tensor([[97, 98]] * 4))


def test_prediction_is_the_row_whose_ending_the_model_expects():
    tokens, mask, _ = hellaswag.render_example(example(2), encode_bytes)
    vocab = 256
    logits = torch.zeros(4, tokens.shape[1], vocab)
    for position in range(tokens.shape[1] - 1):
        logits[2, position, tokens[2, position + 1]] = 10.0

    assert hellaswag.predict(logits, tokens, mask) == 2


def test_load_examples_reads_one_json_object_per_line(tmp_path):
    path = tmp_path / "hellaswag.jsonl"
    path.write_text(json.dumps(example(0)) + "\n" + json.dumps(example(3)) + "\n")

    loaded = hellaswag.load_examples(path)

    assert [item["label"] for item in loaded] == [0, 3]


class FirstEndingModel:
    """Expects the first ending of every example, whatever the example says."""

    def __call__(self, tokens):
        logits = torch.zeros(tokens.size(0), tokens.size(1), 256)
        for row in range(0, tokens.size(0), 4):
            for position in range(tokens.size(1) - 1):
                logits[row, position, tokens[row, position + 1]] = 10.0
        return logits, None


def test_accuracy_counts_the_examples_whose_first_ending_is_correct_across_batches():
    # Ten examples need two batches, so the batching is exercised too.
    examples = [example(label) for label in [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]

    accuracy = hellaswag.accuracy(FirstEndingModel(), examples, "cpu", encode_bytes)

    assert accuracy == 0.5
