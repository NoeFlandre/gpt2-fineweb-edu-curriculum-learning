import json

import torch
from gpt2_scratch.evaluation import multiple_choice as mc


def encode_bytes(text):
    return list(text.encode())


def example(label, endings=("cd", "efg", "h", "ij")):
    return {"ctx": "ab", "endings": list(endings), "label": label}


class CorrectEndingsModel:
    """Makes every row whose text ends with a correct ending the most likely one."""

    def __init__(self, correct_endings):
        self.correct = [ending.encode() for ending in correct_endings]

    def __call__(self, tokens):
        logits = torch.zeros(tokens.size(0), tokens.size(1), 256)
        for row in range(tokens.size(0)):
            text = bytes(int(t) for t in tokens[row] if t > 0)
            if any(text.endswith(ending) for ending in self.correct):
                logits[row, :-1] = torch.nn.functional.one_hot(tokens[row, 1:], 256).float() * 10
        return logits, None


def test_each_ending_is_scored_on_its_own_tokens_after_the_context():
    tokens, mask, label = mc.render_example(example(1), encode_bytes)

    assert tokens.shape[0] == 4
    assert mask[0].sum().item() == len(" cd")
    assert mask[1].sum().item() == len(" efg")
    assert mask[:, :2].sum().item() == 0
    assert label == 1


def test_an_example_keeps_its_own_number_of_endings():
    tokens, _, _ = mc.render_example(example(0, endings=("a", "b")), encode_bytes)

    assert tokens.shape[0] == 2


def test_load_examples_reads_one_json_object_per_line(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(json.dumps(example(0)) + "\n" + json.dumps(example(3)) + "\n")

    loaded = mc.load_examples(path)

    assert [item["label"] for item in loaded] == [0, 3]


def test_accuracy_counts_the_examples_whose_correct_ending_is_most_likely():
    examples = [
        {"ctx": "ab", "endings": ["cd", "efg"], "label": 0},
        {"ctx": "ab", "endings": ["xy", "zw", "qr", "st"], "label": 2},
    ]
    model = CorrectEndingsModel(["cd", "qr"])

    assert mc.accuracy(model, examples, "cpu", encode_bytes) == 1.0


def test_accuracy_is_lower_when_the_label_is_not_the_correct_ending():
    examples = [
        {"ctx": "ab", "endings": ["cd", "efg"], "label": 0},
        {"ctx": "ab", "endings": ["xy", "zw", "qr", "st"], "label": 1},
    ]
    model = CorrectEndingsModel(["cd", "qr"])

    assert mc.accuracy(model, examples, "cpu", encode_bytes) == 0.5


def test_padding_never_wins_when_examples_have_different_numbers_of_endings():
    examples = [
        {"ctx": "ab", "endings": ["cd", "ef"], "label": 0},
        {"ctx": "ab", "endings": ["gh", "ij", "kl", "mn"], "label": 0},
    ]
    model = CorrectEndingsModel(["cd", "ij"])

    assert mc.accuracy(model, examples, "cpu", encode_bytes) == 0.5


def test_batch_size_does_not_change_the_accuracy():
    examples = [{"ctx": "ab", "endings": ["cd", "efg"], "label": index % 2} for index in range(10)]
    model = CorrectEndingsModel(["cd"])

    small = mc.accuracy(model, examples, "cpu", encode_bytes, batch_size=1)
    large = mc.accuracy(model, examples, "cpu", encode_bytes, batch_size=8)

    assert small == large == 0.5
