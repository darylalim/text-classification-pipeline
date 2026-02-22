from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from streamlit_app import BATCH_SIZE, get_device, process_dataframe


# --- BATCH_SIZE ---


def test_batch_size_is_positive_int():
    assert isinstance(BATCH_SIZE, int)
    assert BATCH_SIZE > 0


# --- get_device ---


class TestGetDevice:
    @patch("streamlit_app.torch")
    def test_prefers_mps(self, mock_torch):
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = True
        assert get_device() == "mps"

    @patch("streamlit_app.torch")
    def test_falls_back_to_cuda(self, mock_torch):
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        assert get_device() == "cuda"

    @patch("streamlit_app.torch")
    def test_falls_back_to_cpu(self, mock_torch):
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        assert get_device() == "cpu"


# --- load_model ---


class TestLoadModel:
    @patch.dict("os.environ", {"HF_TOKEN": "test-token"})
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoModelForSequenceClassification")
    def test_loads_correct_model(self, mock_model_cls, mock_tok_cls):
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        from streamlit_app import load_model

        load_model.clear()
        load_model("cpu")

        mock_model_cls.from_pretrained.assert_called_once_with(
            "siebert/sentiment-roberta-large-english",
            dtype=torch.float16,
            token="test-token",
        )
        mock_tok_cls.from_pretrained.assert_called_once_with(
            "siebert/sentiment-roberta-large-english", token="test-token"
        )

    @patch.dict("os.environ", {}, clear=True)
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoModelForSequenceClassification")
    def test_loads_without_token(self, mock_model_cls, mock_tok_cls):
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        from streamlit_app import load_model

        load_model.clear()
        load_model("cpu")

        mock_model_cls.from_pretrained.assert_called_once_with(
            "siebert/sentiment-roberta-large-english",
            dtype=torch.float16,
            token=None,
        )
        mock_tok_cls.from_pretrained.assert_called_once_with(
            "siebert/sentiment-roberta-large-english", token=None
        )

    @patch("streamlit_app.hf_logging")
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoModelForSequenceClassification")
    def test_suppresses_hf_warnings(
        self, mock_model_cls, mock_tok_cls, mock_hf_logging
    ):
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        from streamlit_app import load_model

        load_model.clear()
        load_model("cpu")

        mock_hf_logging.set_verbosity_error.assert_called_once()

    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoModelForSequenceClassification")
    def test_returns_model_and_tokenizer(self, mock_model_cls, mock_tok_cls):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_cls.from_pretrained.return_value.to.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        from streamlit_app import load_model

        load_model.clear()
        model, tokenizer = load_model("cpu")

        assert model is mock_model
        assert tokenizer is mock_tokenizer


# --- process_dataframe ---


def _make_mock_tokenizer():
    """Create a mock tokenizer for sequence classification."""
    tokenizer = MagicMock()

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    tokenizer.return_value = mock_inputs

    return tokenizer


def _make_mock_model(sentiments):
    """Create a mock model that returns logits for the given sentiments.

    Args:
        sentiments: list of "positive" or "negative" strings.
            Each maps to logits where the higher value is at the correct index.
    """
    model = MagicMock()
    model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}

    # Build logits: NEGATIVE=index 0, POSITIVE=index 1
    logits_rows = []
    for s in sentiments:
        if s == "positive":
            logits_rows.append([0.0, 1.0])
        else:
            logits_rows.append([1.0, 0.0])

    mock_output = MagicMock()
    mock_output.logits = torch.tensor(logits_rows)
    model.return_value = mock_output

    return model


class TestProcessDataframe:
    @patch("streamlit_app.st")
    def test_adds_sentiment_column(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["good product", "bad product"]})
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(["positive", "negative"])

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert "Sentiment" in result.columns
        assert len(result) == 2

    @patch("streamlit_app.st")
    def test_classifies_positive(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["great"]})
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(["positive"])

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert result["Sentiment"].iloc[0] == "positive"

    @patch("streamlit_app.st")
    def test_classifies_negative(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["terrible"]})
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(["negative"])

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert result["Sentiment"].iloc[0] == "negative"

    @patch("streamlit_app.st")
    def test_maps_labels_to_lowercase(self, mock_st):
        """Model returns uppercase labels; they should be lowercased."""
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["great", "awful"]})
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(["positive", "negative"])

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert result["Sentiment"].iloc[0] == "positive"
        assert result["Sentiment"].iloc[1] == "negative"

    @patch("streamlit_app.st")
    def test_batching_multiple_batches(self, mock_st):
        """Texts exceeding BATCH_SIZE should be split into multiple batches."""
        mock_st.progress.return_value = MagicMock()
        n = BATCH_SIZE + 3
        df = pd.DataFrame({"text": [f"review {i}" for i in range(n)]})

        model = MagicMock()
        model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}

        # First call: full batch, second call: remaining 3
        batch1_output = MagicMock()
        batch1_output.logits = torch.tensor([[0.0, 1.0]] * BATCH_SIZE)
        batch2_output = MagicMock()
        batch2_output.logits = torch.tensor([[0.0, 1.0]] * 3)
        model.side_effect = [batch1_output, batch2_output]

        tokenizer = _make_mock_tokenizer()

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert len(result) == n
        assert model.call_count == 2

    @patch("streamlit_app.st")
    def test_progress_bar_reaches_completion(self, mock_st):
        mock_progress = MagicMock()
        mock_st.progress.return_value = mock_progress
        df = pd.DataFrame({"text": ["review"]})
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(["positive"])

        process_dataframe(df, "text", model, tokenizer, "cpu")

        last_call_arg = mock_progress.progress.call_args_list[-1][0][0]
        assert last_call_arg == pytest.approx(1.0)

    @patch("streamlit_app.st")
    def test_uses_correct_text_column(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"col_a": ["ignore"], "col_b": ["use this"]})
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(["positive"])

        process_dataframe(df, "col_b", model, tokenizer, "cpu")

        # Tokenizer should be called with the text from col_b
        call_args = tokenizer.call_args
        assert "use this" in call_args[0][0]

    @patch("streamlit_app.st")
    def test_uses_inference_mode(self, mock_st):
        """Verify torch.inference_mode is used (not no_grad)."""
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["test"]})
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(["positive"])

        with patch("streamlit_app.torch.inference_mode") as mock_inf:
            mock_inf.return_value.__enter__ = MagicMock()
            mock_inf.return_value.__exit__ = MagicMock(return_value=False)
            process_dataframe(df, "text", model, tokenizer, "cpu")
            mock_inf.assert_called_once()

    @patch("streamlit_app.st")
    def test_does_not_mutate_input_dataframe(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["review"]})
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(["positive"])

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert "Sentiment" not in df.columns
        assert "Sentiment" in result.columns

    @patch("streamlit_app.st")
    def test_handles_empty_dataframe(self, mock_st):
        mock_progress = MagicMock()
        mock_st.progress.return_value = mock_progress
        df = pd.DataFrame({"text": []})
        model = MagicMock()
        tokenizer = MagicMock()

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert "Sentiment" in result.columns
        assert "Confidence" in result.columns
        assert len(result) == 0
        model.assert_not_called()
        mock_progress.progress.assert_called_once_with(1.0)

    @patch("streamlit_app.st")
    def test_tokenizer_called_with_truncation(self, mock_st):
        """Tokenizer should use truncation=True for RoBERTa's 512 token limit."""
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["a review"]})
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(["positive"])

        process_dataframe(df, "text", model, tokenizer, "cpu")

        call_kwargs = tokenizer.call_args[1]
        assert call_kwargs["truncation"] is True

    @patch("streamlit_app.st")
    def test_tokenizer_called_with_padding(self, mock_st):
        """Tokenizer should use padding=True for batched inputs."""
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["a review"]})
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(["positive"])

        process_dataframe(df, "text", model, tokenizer, "cpu")

        call_kwargs = tokenizer.call_args[1]
        assert call_kwargs["padding"] is True

    @patch("streamlit_app.st")
    def test_uses_id2label_mapping(self, mock_st):
        """Labels should come from model.config.id2label."""
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["review"]})
        tokenizer = _make_mock_tokenizer()

        model = MagicMock()
        model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.0, 1.0]])
        model.return_value = mock_output

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert result["Sentiment"].iloc[0] == "positive"

    @patch("streamlit_app.st")
    def test_adds_confidence_column(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["good product", "bad product"]})
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(["positive", "negative"])

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert "Confidence" in result.columns
        assert len(result["Confidence"]) == 2
        for val in result["Confidence"]:
            assert 0.0 <= val <= 1.0

    @patch("streamlit_app.st")
    def test_handles_all_blank_texts(self, mock_st):
        mock_progress = MagicMock()
        mock_st.progress.return_value = mock_progress
        df = pd.DataFrame({"text": ["", "  ", "\t"]})
        model = MagicMock()
        tokenizer = MagicMock()

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert len(result) == 3
        assert all(s == "" for s in result["Sentiment"])
        assert all(c == 0.0 for c in result["Confidence"])
        model.assert_not_called()
        mock_progress.progress.assert_called_once_with(1.0)

    @patch("streamlit_app.st")
    def test_handles_mixed_blank_text(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["good product", "", "  ", "bad product"]})
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(["positive", "negative"])

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert result["Sentiment"].iloc[1] == ""
        assert result["Confidence"].iloc[1] == 0.0
        assert result["Sentiment"].iloc[2] == ""
        assert result["Confidence"].iloc[2] == 0.0
        assert result["Sentiment"].iloc[0] == "positive"
        assert result["Sentiment"].iloc[3] == "negative"

    @patch("streamlit_app.st")
    def test_confidence_values_in_valid_range(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": [f"text {i}" for i in range(5)]})
        tokenizer = _make_mock_tokenizer()
        model = _make_mock_model(
            ["positive", "negative", "positive", "negative", "positive"]
        )

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        for val in result["Confidence"]:
            assert 0.0 <= val <= 1.0
