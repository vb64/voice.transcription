"""Lib torch tests.

make test T=test_torch.py
"""
import torch

from . import TestBase


class TestTorch(TestBase):
    """Lib torch."""

    def test_types(self):
        """Check torch types."""
        # print(repr(torch.float16))
        assert torch.float16
