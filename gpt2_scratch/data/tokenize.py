"""Turn raw text into GPT-2 token ids."""

import numpy as np
import tiktoken


def encode_document(text: str, encoder: tiktoken.Encoding) -> np.ndarray:
    """Encode one document, prefixed by the end-of-text token that separates documents.

    Args:
        text: The raw document.
        encoder: The GPT-2 tiktoken encoding.

    Returns:
        The token ids as a uint16 array.
    """
    ids = [encoder.eot_token, *encoder.encode_ordinary(text)]
    return np.asarray(ids, dtype=np.uint16)
