"""Execution trace visualization component for Alethia UI.

Presents complete system diagnostics in a collapsed 'Advanced execution details' expander.
Uses robust Markdown and Streamlit components to guarantee zero raw HTML leakage.
"""

from typing import Any, Dict, List, Optional
import streamlit as st


def _get_verdict_badge(verdict: str, is_abstained: bool = False) -> str:
    """Return styled HTML badge for verdict."""
    if is_abstained:
        return '<span class="badge badge-abstained">Abstained · Pass</span>'
    v_upper = (verdict or "").strip().upper()
    if v_upper == "PASS":
        return '<span class="badge badge-pass">✓ Passed</span>'
    elif v_upper == "FAIL":
        return '<span class="badge badge-fail">✗ Failed</span>'
    return f'<span class="badge badge-info">{verdict}</span>'


_get_verdict_label = _get_verdict_badge


def render_trace(
    trace: Optional[List[Dict[str, Any]]],
    original_question: str = "",
    final_question: str = "",
    verdict: str = "",
    attempts: int = 0,
    draft: str = "",
    decomposed_questions: Optional[List[str]] = None,
) -> None:
    """Render an expandable execution details panel showing graph state transitions.

    Args:
        trace: List of chronological trace event dictionaries from graph state.
        original_question: The initial question submitted by the user.
        final_question: The final question used in generation (may be rewritten).
        verdict: Final evaluation verdict ('PASS' or 'FAIL').
        attempts: Number of rewrite attempts executed.
        draft: The generated answer draft (used to detect valid abstention).
        decomposed_questions: Optional list of sub-questions from question decomposition.
    """
    if not trace:
        with st.expander("🔍 Advanced execution details", expanded=False):
            st.info("No execution trace available.")
        return

    is_abstained = "not in context" in (draft or "").lower() and (verdict or "").upper() == "PASS"
    total_steps = len(trace)

    gen_count = sum(1 for e in trace if e.get("node") == "generate_node")
    rewrite_count = sum(1 for e in trace if e.get("node") == "rewrite_node")

    # Find total latency if recorded in completion event
    latency_ms = None
    for e in trace:
        if e.get("event") == "workflow_completed" and "total_latency_ms" in e:
            latency_ms = e["total_latency_ms"]

    verdict_label = _get_verdict_label(verdict, is_abstained)

    with st.expander(f"🔍 Advanced execution details ({total_steps} steps · {attempts} retries)", expanded=False):
        # 1. Summary Header Box using Markdown (zero raw HTML tags)
        summary_md = [
            f"**Execution Summary**",
            f"- **Verdict:** `{verdict_label}`",
            f"- **Original Question:** {original_question}",
        ]
        if final_question and final_question != original_question:
            summary_md.append(f"- **Final Question:** {final_question}")
        if decomposed_questions and len(decomposed_questions) > 1:
            decomp_str = ", ".join(f"`{sq}`" for sq in decomposed_questions)
            summary_md.append(f"- **Decomposed Sub-questions:** {decomp_str}")
        summary_md.append(f"- **Rewrite Attempts:** {attempts} / 2 max")
        summary_md.append(f"- **Generation Cycles:** {gen_count}")
        summary_md.append(f"- **Total Trace Steps:** {total_steps}")
        if latency_ms is not None:
            summary_md.append(f"- **Total Latency:** {latency_ms:.1f} ms")

        st.markdown("\n".join(summary_md))
        st.markdown("---")

        # 2. Chronological Timeline of Steps
        st.markdown("**State Transition Timeline:**")

        for idx, entry in enumerate(trace, 1):
            node = entry.get("node") or entry.get("event", "unknown")

            if node == "retrieve_node":
                chunk_count = entry.get("chunk_count", 0)
                sources = entry.get("sources", [])
                query = entry.get("question", "")
                attempt_num = entry.get("attempt", 0)
                newly_ids = entry.get("newly_retrieved_chunk_ids", [])
                sources_str = ", ".join(sources) if sources else "None"
                step_title = "Initial Retrieval" if attempt_num == 0 else f"Targeted Re-retrieval (Attempt #{attempt_num})"

                st.markdown(
                    f"**Step {idx}: {step_title}**  \n"
                    f"*Query:* `{query}`  \n"
                    f"*Retrieved Chunks:* {chunk_count} · *Sources:* {sources_str}"
                    + (f" · *Newly Retrieved:* {len(newly_ids)} chunks" if newly_ids else "")
                )

            elif node == "generate_node":
                chunk_count = entry.get("chunk_count", 0)
                d = entry.get("draft", "")
                note = entry.get("note")
                preview = d[:180] + ("..." if len(d) > 180 else "")

                st.markdown(
                    f"**Step {idx}: Answer Synthesis**  \n"
                    f"*Context Chunks:* {chunk_count}  \n"
                    + (f"*Note:* {note}  \n" if note else "")
                    + f"*Draft Preview:* {preview}"
                )

            elif node == "judge_node":
                v = entry.get("verdict", "")
                reason = entry.get("reason", "")
                badge_str = "✓ PASS" if v.upper() == "PASS" else "✗ FAIL"

                st.markdown(
                    f"**Step {idx}: Factual Verification ({badge_str})**  \n"
                    f"*Assessment:* {reason}"
                )

            elif node == "rewrite_node":
                att = entry.get("attempt", 0)
                rewritten_q = entry.get("rewritten_question", "")
                verif_reason = entry.get("verification_reason", "")

                st.markdown(
                    f"**Step {idx}: Targeted Query Rewriting (Attempt #{att})**  \n"
                    f"*Targeted Query:* `{rewritten_q}`  \n"
                    f"*Feedback Addressed:* {verif_reason}"
                )

            elif node == "workflow_completed":
                continue

            if idx < len(trace):
                st.caption("↓")
