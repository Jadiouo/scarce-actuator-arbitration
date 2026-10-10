import importlib.util
import os

import numpy as np
import pytest

_p = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "d2_s1_diag_offknob.py")
_spec = importlib.util.spec_from_file_location("d2diag", _p)
d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d)


def test_splits_refuse_test():
    assert len(d.split_seeds("val")) == 32 and len(d.split_seeds("val2")) == 32
    for bad in ("test", "train", "replay"):
        with pytest.raises(ValueError):
            d.split_seeds(bad)


def test_paired_E():
    e = d.paired_E([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert e["mean"] == 0.0 and e["half"] == 0.0
    assert d.paired_E([2, 3, 4, 5], [1, 2, 3, 4])["lo"] == 1.0
