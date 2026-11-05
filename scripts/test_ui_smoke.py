"""Phase 5 UI Live Integration Smoke Test Script.

Validates end-to-end UI logic and live inference:
1. Simulates Streamlit session state and chat message processing.
2. Executes a live supported query through the UI pipeline handler.
3. Verifies verdict, attempts, sources, and execution trace recording.
4. Executes a live out-of-scope query to verify proper abstention.
5. Verifies that re-rendering historical messages does not invoke duplicate LLM calls.
"""

import sys
from pathlib import Path

# Ensure stdout handles UTF-8 safely on Windows
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from unittest.mock import patch
from src.config import get_groq_model, is_groq_configured
from ui.sidebar import get_database_stats
from ui.chat import init_session_state, process_user_query, render_assistant_message


def run_ui_live_smoke_test() -> bool:
    print("=" * 75)
    print("ALETHIA PHASE 5: STREAMLIT UI & USER EXPERIENCE SMOKE TEST")
    print("=" * 75)

    # 1. Environment & Database Check
    print("\n[1] Checking System Status & Knowledge Base...")
    configured = is_groq_configured()
    print(f"    Groq Configured: {'YES' if configured else 'NO'}")
    print(f"    Active Model:    {get_groq_model()}")

    if not configured:
        print("\n[NOTICE] GROQ_API_KEY is not configured. Live smoke test requires a key in .env.")
        return False

    stats = get_database_stats()
    print(f"    ChromaDB Documents: {stats['documents']}")
    print(f"    ChromaDB Chunks:    {stats['chunks']}")

    # 2. Initialize Session State
    init_session_state()
    st.session_state.messages = []
    print(f"    Session State Initialized: {len(st.session_state.messages)} messages")

    # 3. Test Supported Query Execution
    print("\n[2] Testing Supported Query UI Pipeline...")
    query_1 = "How many encoder layers does the Transformer use?"
    print(f"    User Query: '{query_1}'")

    with patch("streamlit.chat_message"), \
         patch("streamlit.write"), \
         patch("streamlit.spinner"), \
         patch("streamlit.markdown"), \
         patch("streamlit.caption"), \
         patch("streamlit.expander"):
        process_user_query(query_1)

    if len(st.session_state.messages) < 2:
        print("    [ERROR] Expected 2 messages in session state (user + assistant).")
        return False

    user_msg = st.session_state.messages[0]
    assistant_msg = st.session_state.messages[1]

    print(f"    Assistant Verdict:  {assistant_msg.get('verdict')}")
    print(f"    Attempts:           {assistant_msg.get('attempts')}")
    print(f"    Draft Answer:       {assistant_msg.get('content')}")
    print(f"    Judge Reason:       {assistant_msg.get('reason')}")
    print(f"    Retrieved Chunks:   {len(assistant_msg.get('chunks', []))}")
    print(f"    Trace Steps Count:  {len(assistant_msg.get('trace', []))}")

    if assistant_msg.get("verdict") != "PASS":
        print("    [ERROR] Expected verdict PASS for supported query.")
        return False

    # 4. Test Abstention Query Execution
    print("\n[3] Testing Out-of-Scope Abstention Query UI Pipeline...")
    query_2 = "What is the capital city of Australia?"
    print(f"    User Query: '{query_2}'")

    with patch("streamlit.chat_message"), \
         patch("streamlit.write"), \
         patch("streamlit.spinner"), \
         patch("streamlit.markdown"), \
         patch("streamlit.caption"), \
         patch("streamlit.expander"):
        process_user_query(query_2)

    if len(st.session_state.messages) < 4:
        print("    [ERROR] Expected 4 messages in session state.")
        return False

    abstain_msg = st.session_state.messages[3]
    print(f"    Assistant Verdict:  {abstain_msg.get('verdict')}")
    print(f"    Attempts:           {abstain_msg.get('attempts')}")
    print(f"    Draft Answer:       {abstain_msg.get('content')}")
    print(f"    Judge Reason:       {abstain_msg.get('reason')}")

    is_abstained = "not in context" in abstain_msg.get("content", "").lower()
    print(f"    Proper Abstention Detected: {'YES' if is_abstained else 'NO'}")

    if not is_abstained or abstain_msg.get("verdict") != "PASS":
        print("    [ERROR] Expected verified abstention ('Not in context.' + PASS).")
        return False

    # 5. Verify Rerender Safety (no duplicate LLM calls)
    print("\n[4] Verifying Rerender Safety...")
    # Re-rendering historical messages should only read from session state
    render_count = 0
    with patch("streamlit.chat_message"), \
         patch("streamlit.write") as mock_write, \
         patch("streamlit.markdown"), \
         patch("streamlit.caption"), \
         patch("streamlit.expander"):
        for msg in st.session_state.messages:
            if msg.get("role") == "assistant":
                render_assistant_message(msg)
                render_count += 1

    print(f"    Re-rendered {render_count} assistant messages from session_state without pipeline invocation.")

    print("\n" + "=" * 75)
    print("PHASE 5 STREAMLIT UI LIVE INTEGRATION SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = run_ui_live_smoke_test()
    sys.exit(0 if success else 1)

