"""Modern, professional SaaS theme styling and custom CSS for Alethia.

Implements a calm, clean, high-contrast palette:
- Main background: #F8FAFC
- Main surface: #FFFFFF
- Primary text: #172033
- Secondary text: #64748B
- Primary accent: #4F46E5
- Subtle accent: #EEF2FF
- Success: #15803D
- Warning: #B45309
- Error: #B91C1C
- Borders: #E2E8F0
"""

import streamlit as st


def get_custom_css() -> str:
    """Return polished custom CSS string for modern, accessible SaaS UI."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #172033;
    }

    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .block-container {
        max-width: 920px;
        padding-top: 1.75rem;
        padding-bottom: 4rem;
    }

    /* Header styling */
    .alethia-header {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #E2E8F0;
    }

    .alethia-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #172033;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .alethia-subtitle {
        font-size: 0.925rem;
        color: #64748B;
        margin-top: 0.25rem;
    }

    /* Clean User-Facing Status Badges */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.775rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .status-verified {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }

    .status-partial {
        background: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
    }

    .status-notfound {
        background: #F1F5F9;
        color: #64748B;
        border: 1px solid #E2E8F0;
    }

    .status-error {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }

    /* Welcome card */
    .welcome-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    .welcome-card h3 {
        margin-top: 0;
        color: #172033;
        font-size: 1.2rem;
        font-weight: 600;
    }

    .welcome-card p {
        color: #64748B;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }

    /* Source chunk cards */
    .source-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    .source-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.775rem;
        color: #64748B;
        margin-bottom: 0.45rem;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 0.35rem;
    }

    .source-text {
        font-size: 0.85rem;
        color: #334155;
        line-height: 1.55;
        white-space: pre-wrap;
    }

    /* Document item card in sidebar */
    .doc-item-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.5rem;
    }

    .doc-item-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.825rem;
        font-weight: 600;
        color: #172033;
    }

    .doc-item-sub {
        font-size: 0.725rem;
        color: #64748B;
        margin-top: 0.2rem;
    }

    /* Clean diagnostic box */
    .diagnostic-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Chat input override to eliminate harsh red borders */
    .stChatInput > div {
        border-color: #CBD5E1 !important;
        border-radius: 0.5rem !important;
    }
    .stChatInput > div:focus-within {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 1px #4F46E5 !important;
    }
    </style>
    """


def apply_custom_styles() -> None:
    """Inject custom SaaS theme styling into the active Streamlit app."""
    st.markdown(get_custom_css(), unsafe_allow_html=True)
