import sys
import os

# Add the parent directory to the Python path
sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

from main import train_model  # noqa: E402


def test_train_model():
    model, acc = train_model()
    assert model is not None
    assert acc > 0.5
