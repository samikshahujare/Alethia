"""Modern, minimalist luxury theme styling and custom CSS for Alethia.

Implements a dark-first, highly authentic, professional palette with clean typography,
seamless theme adaptability, and zero cluttered or cartoonish elements.
"""

import streamlit as st


def get_custom_css() -> str:
    """Return polished custom CSS string for modern, accessible SaaS UI."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        letter-spacing: -0.01em;
    }

    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .block-container {
        max-width: 940px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    /* Minimalist header */
    .alethia-header {
        margin-bottom: 1.75rem;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    }

    .alethia-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }

    .header-engine-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        background: rgba(99, 102, 241, 0.1);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }

    .alethia-subtitle {
        font-size: 0.88rem;
        color: #94A3B8;
        margin-top: 0.35rem;
        line-height: 1.5;
    }

    /* Authentic, Understated Status Badges */
    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.65rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .status-verified {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.28);
    }

    .status-partial {
        background: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.28);
    }

    .status-notfound {
        background: rgba(100, 116, 139, 0.12);
        color: #94A3B8;
        border: 1px solid rgba(100, 116, 139, 0.25);
    }

    .status-error {
        background: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.28);
    }

    /* Diagnostics Verdict Badges */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .badge-pass {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-fail {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .badge-abstained {
        background: rgba(100, 116, 139, 0.15);
        color: #94A3B8;
        border: 1px solid rgba(100, 116, 139, 0.3);
    }

    .badge-info {
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Welcome card */
    .welcome-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 0.65rem;
        padding: 1.5rem;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }

    /* Source chunk cards */
    .source-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 0.5rem;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }

    .source-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.775rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.08);
        padding-bottom: 0.4rem;
    }

    .file-type-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        padding: 0.12rem 0.4rem;
        border-radius: 3px;
        background: rgba(99, 102, 241, 0.12);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }

    .source-score-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.12rem 0.45rem;
        border-radius: 3px;
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .source-text {
        font-size: 0.85rem;
        line-height: 1.6;
        white-space: pre-wrap;
        opacity: 0.9;
    }

    /* Sidebar minimal branding and components */
    .sidebar-brand-box {
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    }

    .sidebar-brand-title {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: inherit;
    }

    .sidebar-brand-sub {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 0.15rem;
        letter-spacing: 0.02em;
    }

    .sidebar-divider {
        height: 1px;
        background: rgba(148, 163, 184, 0.12);
        margin: 1.25rem 0;
    }

    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 0.75rem;
    }

    .sidebar-doc-row {
        overflow: hidden;
    }

    .sidebar-doc-name {
        font-size: 0.825rem;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .sidebar-doc-meta {
        font-size: 0.72rem;
        color: #94A3B8;
        margin-top: 0.15rem;
    }

    /* Knowledge base metrics card */
    .kb-metrics-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 0.5rem;
        padding: 0.75rem 0.85rem;
        margin-bottom: 0.75rem;
    }

    .kb-metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        padding: 0.25rem 0;
        color: #94A3B8;
    }

    .kb-metric-row strong {
        color: inherit;
    }

    /* Streamlit interactive polish */
    .stChatInput > div {
        border-color: rgba(148, 163, 184, 0.2) !important;
        border-radius: 0.5rem !important;
    }
    .stChatInput > div:focus-within {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 1px #6366F1 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.4rem 0.8rem;
        font-size: 0.8rem;
        border-radius: 4px;
    }
    </style>
    """


def apply_custom_styles() -> None:
    """Inject custom SaaS theme styling into the active Streamlit app."""
    st.markdown(get_custom_css(), unsafe_allow_html=True)
