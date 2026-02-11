from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from streamlit_app import BATCH_SIZE, PROMPT_TEMPLATE, get_device, process_dataframe


# --- PROMPT_TEMPLATE ---


class TestPromptTemplate:
    def test_contains_placeholder(self):
        assert "{}" in PROMPT_TEMPLATE

    def test_format_inserts_review(self):
        result = PROMPT_TEMPLATE.format("Great product!")
        assert "Great product!" in result
        assert "{}" not in result

    def test_contains_few_shot_examples(self):
        assert "positive" in PROMPT_TEMPLATE
        assert "negative" in PROMPT_TEMPLATE

    def test_ends_with_sentiment_label(self):
        assert PROMPT_TEMPLATE.rstrip().endswith("Sentiment:")


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
    @patch("streamlit_app.AutoModelForCausalLM")
    def test_loads_correct_model(self, mock_model_cls, mock_tok_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        # load_model is wrapped by @st.cache_resource; call the underlying function
        from streamlit_app import load_model

        load_model.clear()
        load_model("cpu")

        mock_model_cls.from_pretrained.assert_called_once_with(
            "ibm-granite/granite-4.0-h-tiny",
            device_map="cpu",
            torch_dtype=torch.float16,
            token="test-token",
        )
        mock_tok_cls.from_pretrained.assert_called_once_with(
            "ibm-granite/granite-4.0-h-tiny", token="test-token"
        )

    @patch.dict("os.environ", {}, clear=True)
    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoModelForCausalLM")
    def test_loads_without_token(self, mock_model_cls, mock_tok_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        from streamlit_app import load_model

        load_model.clear()
        load_model("cpu")

        mock_model_cls.from_pretrained.assert_called_once_with(
            "ibm-granite/granite-4.0-h-tiny",
            device_map="cpu",
            torch_dtype=torch.float16,
            token=None,
        )
        mock_tok_cls.from_pretrained.assert_called_once_with(
            "ibm-granite/granite-4.0-h-tiny", token=None
        )

    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoModelForCausalLM")
    def test_sets_left_padding(self, _mock_model_cls, mock_tok_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "existing"
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        from streamlit_app import load_model

        load_model.clear()
        _, tokenizer = load_model("cpu")

        assert tokenizer.padding_side == "left"

    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoModelForCausalLM")
    def test_assigns_pad_token_when_missing(self, _mock_model_cls, mock_tok_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        from streamlit_app import load_model

        load_model.clear()
        _, tokenizer = load_model("cpu")

        assert tokenizer.pad_token == "<eos>"

    @patch("streamlit_app.AutoTokenizer")
    @patch("streamlit_app.AutoModelForCausalLM")
    def test_keeps_existing_pad_token(self, _mock_model_cls, mock_tok_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        from streamlit_app import load_model

        load_model.clear()
        _, tokenizer = load_model("cpu")

        assert tokenizer.pad_token == "<pad>"


# --- process_dataframe ---


def _make_mock_tokenizer(decode_values):
    """Create a mock tokenizer that returns decode_values in order."""
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = lambda chat, **kw: f"chat:{chat}"

    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = MagicMock(
        side_effect=lambda key: torch.zeros(1, 5) if key == "input_ids" else None
    )
    mock_inputs.to.return_value = mock_inputs
    tokenizer.return_value = mock_inputs

    tokenizer.decode = MagicMock(side_effect=decode_values)
    return tokenizer


def _make_mock_model(batch_size):
    """Create a mock model whose generate returns tensors of the right shape."""
    model = MagicMock()
    model.generate.return_value = torch.zeros(batch_size, 15)
    return model


class TestProcessDataframe:
    @patch("streamlit_app.st")
    def test_adds_sentiment_column(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["good product", "bad product"]})
        tokenizer = _make_mock_tokenizer(["\npositive", "\nnegative"])
        model = _make_mock_model(2)

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert "Sentiment" in result.columns
        assert len(result) == 2

    @patch("streamlit_app.st")
    def test_classifies_positive(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["great"]})
        tokenizer = _make_mock_tokenizer(["\npositive"])
        model = _make_mock_model(1)

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert result["Sentiment"].iloc[0] == "positive"

    @patch("streamlit_app.st")
    def test_classifies_negative(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["terrible"]})
        tokenizer = _make_mock_tokenizer(["\nnegative"])
        model = _make_mock_model(1)

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert result["Sentiment"].iloc[0] == "negative"

    @patch("streamlit_app.st")
    def test_negative_case_insensitive(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["awful"]})
        tokenizer = _make_mock_tokenizer(["\nNegative"])
        model = _make_mock_model(1)

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert result["Sentiment"].iloc[0] == "negative"

    @patch("streamlit_app.st")
    def test_defaults_to_positive(self, mock_st):
        """Unrecognized output should default to positive."""
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["hmm"]})
        tokenizer = _make_mock_tokenizer(["some random text"])
        model = _make_mock_model(1)

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert result["Sentiment"].iloc[0] == "positive"

    @patch("streamlit_app.st")
    def test_batching_multiple_batches(self, mock_st):
        """Reviews exceeding BATCH_SIZE should be split into multiple batches."""
        mock_st.progress.return_value = MagicMock()
        n = BATCH_SIZE + 3
        df = pd.DataFrame({"text": [f"review {i}" for i in range(n)]})

        model = MagicMock()
        # First call: full batch, second call: remaining 3
        model.generate.side_effect = [
            torch.zeros(BATCH_SIZE, 15),
            torch.zeros(3, 15),
        ]

        decode_values = ["\npositive"] * n
        tokenizer = _make_mock_tokenizer(decode_values)

        # Make tokenizer return the right shape for each batch
        def make_inputs(texts, **kwargs):
            mock_inputs = MagicMock()
            mock_inputs.__getitem__ = MagicMock(
                side_effect=lambda key: (
                    torch.zeros(len(texts), 5) if key == "input_ids" else None
                )
            )
            mock_inputs.to.return_value = mock_inputs
            return mock_inputs

        tokenizer.side_effect = make_inputs

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert len(result) == n
        assert model.generate.call_count == 2

    @patch("streamlit_app.st")
    def test_progress_bar_reaches_completion(self, mock_st):
        mock_progress = MagicMock()
        mock_st.progress.return_value = mock_progress
        df = pd.DataFrame({"text": ["review"]})
        tokenizer = _make_mock_tokenizer(["\npositive"])
        model = _make_mock_model(1)

        process_dataframe(df, "text", model, tokenizer, "cpu")

        # Last progress call should be 1.0 (complete)
        last_call_arg = mock_progress.progress.call_args_list[-1][0][0]
        assert last_call_arg == pytest.approx(1.0)

    @patch("streamlit_app.st")
    def test_uses_correct_text_column(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"col_a": ["ignore"], "col_b": ["use this"]})
        tokenizer = _make_mock_tokenizer(["\npositive"])
        model = _make_mock_model(1)

        process_dataframe(df, "col_b", model, tokenizer, "cpu")

        # The prompt should contain the text from col_b
        chat_arg = tokenizer.apply_chat_template.call_args[0][0]
        assert "use this" in chat_arg[0]["content"]

    @patch("streamlit_app.st")
    def test_uses_inference_mode(self, mock_st):
        """Verify torch.inference_mode is used (not no_grad)."""
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["test"]})
        tokenizer = _make_mock_tokenizer(["\npositive"])
        model = _make_mock_model(1)

        with patch("streamlit_app.torch.inference_mode") as mock_inf:
            mock_inf.return_value.__enter__ = MagicMock()
            mock_inf.return_value.__exit__ = MagicMock(return_value=False)
            process_dataframe(df, "text", model, tokenizer, "cpu")
            mock_inf.assert_called_once()

    @patch("streamlit_app.st")
    def test_does_not_mutate_input_dataframe(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["review"]})
        tokenizer = _make_mock_tokenizer(["\npositive"])
        model = _make_mock_model(1)

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
        assert len(result) == 0
        model.generate.assert_not_called()
        mock_progress.progress.assert_called_once_with(1.0)

    @patch("streamlit_app.st")
    def test_handles_curly_braces_in_review(self, mock_st):
        mock_st.progress.return_value = MagicMock()
        df = pd.DataFrame({"text": ["I love {this} product!"]})
        tokenizer = _make_mock_tokenizer(["\npositive"])
        model = _make_mock_model(1)

        result = process_dataframe(df, "text", model, tokenizer, "cpu")

        assert result["Sentiment"].iloc[0] == "positive"
        chat_arg = tokenizer.apply_chat_template.call_args[0][0]
        assert "I love {this} product!" in chat_arg[0]["content"]
