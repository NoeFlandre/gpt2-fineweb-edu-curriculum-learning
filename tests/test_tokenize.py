import numpy as np
import tiktoken
from gpt2_scratch.data.tokenize import encode_document

ENCODER = tiktoken.get_encoding("gpt2")


def test_a_document_starts_with_the_end_of_text_token():
    ids = encode_document("Hello world", ENCODER)

    assert ids[0] == ENCODER.eot_token
    assert ids[1:].tolist() == ENCODER.encode_ordinary("Hello world")


def test_token_ids_are_stored_as_uint16():
    assert encode_document("Hello", ENCODER).dtype == np.uint16
