"""Module language.py tests.

make test T=test_language.py
"""
import pytest

from . import TestBase


class TestLanguage(TestBase):
    """Module language."""

    def test_process_language_arg(self):
        """Check process_language_arg function."""
        from voice_transcription.language import process_language_arg
        from voice_transcription import Model

        assert process_language_arg('ru', Model.Large) == 'ru'
        assert process_language_arg(None, Model.Large) is None
        assert process_language_arg("burmese", Model.Large) == "my"

        with pytest.raises(ValueError) as err:
            process_language_arg('ru', 'large.en')
        assert 'English-only model' in str(err.value)

        with pytest.raises(ValueError) as err:
            process_language_arg('xxx', 'large.en')
        assert 'Unsupported language' in str(err.value)
