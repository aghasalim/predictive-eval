"""Environment probe, not a predictor. Import-time failure means one of these
is unavailable in the scoring container; a finished run means all of them work."""
import numpy  # noqa: F401
import torch  # noqa: F401
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_vec = _model.encode(["probe"])
assert _vec.shape[-1] == 384


def predict(input, labeled=None):
    return 0.5
