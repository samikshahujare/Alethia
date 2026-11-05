"""Unit tests for Phase 3 Groq LLM integration and independent LLM nodes.

Uses mocked Groq responses to guarantee 100% deterministic testing without
consuming live Groq API credits.
"""

from unittest.mock import MagicMock, patch
import pytest
import groq

from src.llm import (
    DEFAULT_MODEL,
    ConfigurationError,
    LLMRateLimitError,
    call_llm,
    get_groq_client,
)
from src.nodes import generate_node, judge_node, rewrite_node
from src.state import GraphState, create_initial_state


class TestGroqClientAndWrapper:
    """Test suite for src.llm client initialization and call_llm wrapper."""

    def test_missing_api_key_raises_configuration_error(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "")
        with pytest.raises(ConfigurationError, match="GROQ_API_KEY is not configured"):
            get_groq_client()

        monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
        with pytest.raises(ConfigurationError, match="GROQ_API_KEY is not configured"):
            get_groq_client()

    def test_call_llm_invokes_correct_model_and_temperature(self):
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response from Groq."
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        messages = [{"role": "user", "content": "Hello"}]
        result = call_llm(
            messages=messages,
            temperature=0.1,
            max_tokens=500,
            model=DEFAULT_MODEL,
            client=mock_client,
        )

        assert result == "Test response from Groq."
        mock_client.chat.completions.create.assert_called_once_with(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=500,
        )

    def test_call_llm_respects_groq_model_env_var(self, monkeypatch):
        monkeypatch.setenv("GROQ_MODEL", "custom/test-model-42b")
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Custom model response."
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        res = call_llm(
            messages=[{"role": "user", "content": "Hi"}],
            client=mock_client,
        )

        assert res == "Custom model response."
        assert mock_client.chat.completions.create.call_args[1]["model"] == "custom/test-model-42b"

    def test_call_llm_handles_rate_limit_error(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = groq.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(),
            body={"error": {"message": "Rate limit reached"}},
        )

        with pytest.raises(LLMRateLimitError, match="rate limit exceeded"):
            call_llm(
                messages=[{"role": "user", "content": "Hi"}],
                client=mock_client,
            )


class TestGenerateNode:
    """Test suite for generate_node."""

    def test_generates_answer_when_context_is_provided(self):
        state: GraphState = {
            "question": "How many layers are in the Transformer?",
            "original_question": "How many layers are in the Transformer?",
            "chunks": [
                {
                    "chunk_id": "c1",
                    "source": "transformer.txt",
                    "text": "The Transformer features six encoder layers and six decoder layers.",
                }
            ],
            "draft": "",
            "trace": [],
        }

        mock_caller = MagicMock(return_value="The Transformer uses six encoder and six decoder layers.")

        updated = generate_node(state, llm_func=mock_caller)

        assert updated["draft"] == "The Transformer uses six encoder and six decoder layers."
        assert updated["question"] == state["question"]
        assert len(updated["chunks"]) == 1
        assert len(updated["trace"]) == 1
        assert updated["trace"][0]["node"] == "generate_node"

        # Verify temperature 0.1 passed to caller
        call_kwargs = mock_caller.call_args[1]
        assert call_kwargs["temperature"] == 0.1

    def test_handles_empty_context_without_calling_llm(self):
        state: GraphState = {
            "question": "What is the capital of Mars?",
            "original_question": "What is the capital of Mars?",
            "chunks": [],
            "draft": "",
            "trace": [],
        }

        mock_caller = MagicMock()
        updated = generate_node(state, llm_func=mock_caller)

        # Must return 'Not in context.' and bypass calling the LLM
        assert updated["draft"] == "Not in context."
        assert not mock_caller.called
        assert updated["trace"][0]["chunk_count"] == 0

    def test_preserves_question_and_chunks(self):
        state = create_initial_state("Explain attention.")
        state["chunks"] = [{"chunk_id": "c1", "text": "Self-attention mechanism."}]

        mock_caller = MagicMock(return_value="Attention computes alignments.")
        updated = generate_node(state, llm_func=mock_caller)

        assert updated["question"] == "Explain attention."
        assert updated["original_question"] == "Explain attention."
        assert updated["chunks"] == state["chunks"]


class TestJudgeNode:
    """Test suite for judge_node."""

    def test_verdict_pass_for_supported_answer(self):
        state: GraphState = {
            "question": "How many encoder layers are there?",
            "chunks": [{"text": "The model has six encoder layers and six decoder layers."}],
            "draft": "There are six encoder layers.",
            "trace": [],
        }

        mock_caller = MagicMock(
            return_value="VERDICT: PASS\nREASON: The context explicitly mentions six encoder layers."
        )

        updated = judge_node(state, llm_func=mock_caller)

        assert updated["verdict"] == "PASS"
        assert "six encoder layers" in updated["reason"]
        assert updated["trace"][-1]["verdict"] == "PASS"
        # Verify judge temperature is strictly 0.0
        assert mock_caller.call_args[1]["temperature"] == 0.0

    def test_verdict_fail_for_unsupported_answer(self):
        state: GraphState = {
            "question": "How many encoder layers are there?",
            "chunks": [{"text": "The model uses multi-head attention."}],
            "draft": "The model uses twenty-four encoder layers.",
            "trace": [],
        }

        mock_caller = MagicMock(
            return_value="VERDICT: FAIL\nREASON: The context does not mention twenty-four layers."
        )

        updated = judge_node(state, llm_func=mock_caller)

        assert updated["verdict"] == "FAIL"
        assert "twenty-four" in updated["reason"]

    def test_verdict_fail_for_contradictory_answer(self):
        state: GraphState = {
            "question": "What optimizer was used?",
            "chunks": [{"text": "Training was conducted using the Adam optimizer."}],
            "draft": "Training was conducted using standard SGD with no momentum.",
            "trace": [],
        }

        mock_caller = MagicMock(
            return_value="VERDICT: FAIL\nREASON: Contradicts the text which specifies the Adam optimizer."
        )

        updated = judge_node(state, llm_func=mock_caller)

        assert updated["verdict"] == "FAIL"
        assert "Contradicts" in updated["reason"]

    def test_handles_not_in_context_draft_appropriately(self):
        # When context is empty and draft is 'Not in context.', it is an accurate faithful statement
        state: GraphState = {
            "question": "What is the airspeed velocity of an unladen swallow?",
            "chunks": [{"text": "The Transformer was proposed in 2017."}],
            "draft": "Not in context.",
            "trace": [],
        }

        mock_caller = MagicMock(
            return_value="VERDICT: PASS\nREASON: The context does not contain the answer, so 'Not in context.' is correct."
        )

        updated = judge_node(state, llm_func=mock_caller)

        assert updated["verdict"] == "PASS"

    def test_handles_malformed_judge_output_safely(self):
        state: GraphState = {
            "question": "Test query",
            "chunks": [{"text": "Some text"}],
            "draft": "Some draft",
            "trace": [],
        }

        # Response missing VERDICT prefix entirely
        mock_caller = MagicMock(return_value="I think the answer seems pretty good and reasonable.")

        updated = judge_node(state, llm_func=mock_caller)

        # Must fail safely
        assert updated["verdict"] == "FAIL"
        assert "failed safely" in updated["reason"].lower()


class TestJudgeHardenedScenarios:
    """Step 4.1: Tests for hardened Judge evaluation across all five required behaviors

    including single-part and multi-part questions.
    """

    def test_scenario_1_fully_grounded_single_part_returns_pass(self):
        """Scenario 1: All required info present, grounded answer -> PASS."""
        state: GraphState = {
            "question": "What optimizer was used to train the Transformer?",
            "chunks": [{"text": "Training was conducted using the Adam optimizer with warmup."}],
            "draft": "The Transformer was trained using the Adam optimizer with warmup.",
            "trace": [],
        }
        mock_caller = MagicMock(
            return_value="VERDICT: PASS\nREASON: All required information is present and the answer is grounded."
        )
        updated = judge_node(state, llm_func=mock_caller)
        assert updated["verdict"] == "PASS"
        assert "grounded" in updated["reason"].lower()

    def test_scenario_1_fully_grounded_multipart_returns_pass(self):
        """Scenario 1: Multi-part question where all parts are present and grounded -> PASS."""
        state: GraphState = {
            "question": "What optimizer was used for the Transformer, and what are Adam's default beta parameters?",
            "chunks": [
                {"text": "Training was conducted using the Adam optimizer with warmup."},
                {"text": "Standard Adam uses beta_1 equals 0.9 and beta_2 equals 0.999."},
            ],
            "draft": "The Transformer used Adam with warmup. Default parameters are beta_1 = 0.9 and beta_2 = 0.999.",
            "trace": [],
        }
        mock_caller = MagicMock(
            return_value="VERDICT: PASS\nREASON: Both parts of the user question are fully supported by the context."
        )
        updated = judge_node(state, llm_func=mock_caller)
        assert updated["verdict"] == "PASS"
        assert "supported" in updated["reason"].lower()

    def test_scenario_2_correct_abstention_missing_info_returns_pass(self):
        """Scenario 2: Required info missing, generator correctly abstains -> PASS."""
        state: GraphState = {
            "question": "What is the capital of Mars?",
            "chunks": [{"text": "Mars hosts Olympus Mons, a volcano 21.9 km high."}],
            "draft": "Not in context.",
            "trace": [],
        }
        mock_caller = MagicMock(
            return_value="VERDICT: PASS\nREASON: Required information is absent and the generator correctly abstains."
        )
        updated = judge_node(state, llm_func=mock_caller)
        assert updated["verdict"] == "PASS"

    def test_scenario_2_correct_abstention_multipart_partial_retrieval_returns_pass(self):
        """Scenario 2: Multi-part question where part B is missing; generator correctly

        abstains on the missing part -> PASS.
        """
        state: GraphState = {
            "question": "What optimizer was used in Attention Is All You Need, and what are standard Adam's beta values?",
            "chunks": [{"text": "Standard Adam uses beta_1 equals 0.9 and beta_2 equals 0.999."}],
            "draft": "Standard Adam uses beta_1 = 0.9 and beta_2 = 0.999. The optimizer used in Attention Is All You Need is not in context.",
            "trace": [],
        }
        mock_caller = MagicMock(
            return_value="VERDICT: PASS\nREASON: The answer provides the grounded beta values and correctly abstains on the missing paper optimizer."
        )
        updated = judge_node(state, llm_func=mock_caller)
        assert updated["verdict"] == "PASS"

    def test_scenario_3_unsupported_claims_missing_info_returns_fail(self):
        """Scenario 3: Required info missing but generator invents unsupported claim -> FAIL."""
        state: GraphState = {
            "question": "What is the airspeed velocity of an unladen swallow?",
            "chunks": [{"text": "The Transformer uses multi-head attention."}],
            "draft": "The airspeed velocity of an unladen swallow is approximately 11 meters per second.",
            "trace": [],
        }
        mock_caller = MagicMock(
            return_value="VERDICT: FAIL\nREASON: The context does not mention swallow velocities; claim is unsupported."
        )
        updated = judge_node(state, llm_func=mock_caller)
        assert updated["verdict"] == "FAIL"
        assert "unsupported" in updated["reason"].lower()

    def test_scenario_3_unsupported_claims_multipart_partial_retrieval_returns_fail(self):
        """Scenario 3: Multi-part question where part A is present but part B is missing;

        generator invents part B from parametric memory -> FAIL.
        """
        state: GraphState = {
            "question": "What are Adam's beta parameters, and what BLEU score did the Transformer get on French translation?",
            "chunks": [{"text": "Standard Adam uses beta_1 equals 0.9 and beta_2 equals 0.999."}],
            "draft": "Adam's beta parameters are beta_1 = 0.9 and beta_2 = 0.999. The Transformer scored 41.8 BLEU on French translation.",
            "trace": [],
        }
        mock_caller = MagicMock(
            return_value="VERDICT: FAIL\nREASON: The French translation BLEU score is not present in the context; claim is unsupported."
        )
        updated = judge_node(state, llm_func=mock_caller)
        assert updated["verdict"] == "FAIL"

    def test_scenario_4_unnecessary_abstention_when_info_present_returns_fail(self):
        """Scenario 4: All required info present, but generator unnecessarily abstains -> FAIL."""
        state: GraphState = {
            "question": "How many encoder layers are in the base Transformer?",
            "chunks": [{"text": "The base architecture contains six encoder layers and six decoder layers."}],
            "draft": "Not in context.",
            "trace": [],
        }
        mock_caller = MagicMock(
            return_value="VERDICT: FAIL\nREASON: The context clearly states the base architecture has six encoder layers; abstention is unnecessary."
        )
        updated = judge_node(state, llm_func=mock_caller)
        assert updated["verdict"] == "FAIL"
        assert "unnecessary" in updated["reason"].lower() or "clearly" in updated["reason"].lower()

    def test_scenario_4_unnecessary_abstention_multipart_all_present_returns_fail(self):
        """Scenario 4: Multi-part question where all parts are present, but generator abstains -> FAIL."""
        state: GraphState = {
            "question": "How many encoder layers are there and what optimizer was used?",
            "chunks": [
                {"text": "The base architecture contains six encoder layers and six decoder layers."},
                {"text": "Training was conducted using the Adam optimizer with warmup."},
            ],
            "draft": "Not in context.",
            "trace": [],
        }
        mock_caller = MagicMock(
            return_value="VERDICT: FAIL\nREASON: All required information is clearly present in context; abstention is unnecessary."
        )
        updated = judge_node(state, llm_func=mock_caller)
        assert updated["verdict"] == "FAIL"

    def test_scenario_5_contradiction_returns_fail(self):
        """Scenario 5: Generated answer directly contradicts retrieved evidence -> FAIL."""
        state: GraphState = {
            "question": "What is the gradient clipping threshold?",
            "chunks": [{"text": "Gradient clipping with a maximum norm threshold of 1.0 restricts gradient magnitudes."}],
            "draft": "The maximum norm threshold for gradient clipping is 50.0.",
            "trace": [],
        }
        mock_caller = MagicMock(
            return_value="VERDICT: FAIL\nREASON: Contradicts the context which specifies a maximum norm threshold of 1.0."
        )
        updated = judge_node(state, llm_func=mock_caller)
        assert updated["verdict"] == "FAIL"
        assert "contradicts" in updated["reason"].lower()

    def test_judge_system_prompt_contains_all_five_rules(self):
        """Confirm that JUDGE_SYSTEM_PROMPT explicitly encodes all 5 required rules."""
        from src.prompts import JUDGE_SYSTEM_PROMPT
        assert "FULLY GROUNDED" in JUDGE_SYSTEM_PROMPT
        assert "CORRECT ABSTENTION" in JUDGE_SYSTEM_PROMPT
        assert "UNSUPPORTED CLAIMS" in JUDGE_SYSTEM_PROMPT
        assert "UNNECESSARY ABSTENTION" in JUDGE_SYSTEM_PROMPT
        assert "CONTRADICTION" in JUDGE_SYSTEM_PROMPT


class TestRewriteNode:
    """Test suite for rewrite_node."""

    def test_produces_rewritten_question_and_preserves_original(self):
        state: GraphState = {
            "question": "layers?",
            "original_question": "layers?",
            "reason": "Query is too vague to retrieve layer counts.",
            "attempts": 0,
            "trace": [],
        }

        mock_caller = MagicMock(
            return_value="How many encoder and decoder layers does the Transformer architecture use?"
        )

        updated = rewrite_node(state, llm_func=mock_caller)

        assert updated["question"] == "How many encoder and decoder layers does the Transformer architecture use?"
        assert updated["original_question"] == "layers?"
        assert updated["attempts"] == 1
        assert len(updated["trace"]) == 1
        assert updated["trace"][0]["node"] == "rewrite_node"

        # Verify rewrite temperature 0.2
        assert mock_caller.call_args[1]["temperature"] == 0.2

    def test_increments_attempts_counter_accurately(self):
        state: GraphState = {
            "question": "current attempt 2",
            "original_question": "initial query",
            "attempts": 2,
            "reason": "Previous context lacked facts.",
            "trace": [],
        }

        mock_caller = MagicMock(return_value="refined query attempt 3")
        updated = rewrite_node(state, llm_func=mock_caller)

        assert updated["attempts"] == 3
        assert updated["original_question"] == "initial query"

    def test_rewrite_does_not_generate_answer_or_perform_retrieval(self):
        state: GraphState = {
            "question": "optimization?",
            "original_question": "optimization?",
            "chunks": [{"text": "Adam optimizer."}],
            "draft": "old draft",
            "trace": [],
        }

        mock_caller = MagicMock(return_value="Which optimization algorithm is used during training?")
        updated = rewrite_node(state, llm_func=mock_caller)

        # Verify draft was NOT altered or generated in this node
        assert updated["draft"] == "old draft"
        # Verify chunks list was untouched
        assert len(updated["chunks"]) == 1
        assert updated["chunks"][0]["text"] == "Adam optimizer."

