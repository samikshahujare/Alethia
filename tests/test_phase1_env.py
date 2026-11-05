"""Phase 1 Environment and Project Foundation Validation Script.

Checks:
1. Python version compatibility (Python 3.10+).
2. Required dependencies importability.
3. Existence of .env.example with GROQ_API_KEY placeholder.
4. Existence and factual accuracy of data/transformer_summary.txt.
5. Presence of complete project directory and module structure.
6. Safe execution of configuration loader without credential exposure.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_python_version() -> tuple[bool, str]:
    """Check if Python version is >= 3.10."""
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro
    version_str = f"{major}.{minor}.{micro}"
    if major == 3 and minor >= 10:
        return True, f"Python version is {version_str} (compatible >= 3.10)"
    return False, f"Python version {version_str} is incompatible (must be >= 3.10)"


def check_required_imports() -> tuple[bool, str]:
    """Check that all required Phase 1 libraries can be imported."""
    packages = {
        "langgraph": "langgraph",
        "groq": "groq",
        "chromadb": "chromadb",
        "sentence_transformers": "sentence-transformers",
        "streamlit": "streamlit",
        "dotenv": "python-dotenv",
    }
    failed = []
    imported_info = []

    for mod, pkg_name in packages.items():
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "unknown")
            imported_info.append(f"{pkg_name} ({ver})")
        except Exception as e:
            failed.append(f"{pkg_name}: {e}")

    if failed:
        return False, f"Failed imports: {', '.join(failed)}"
    return True, f"All packages imported successfully: {', '.join(imported_info)}"


def check_env_example() -> tuple[bool, str]:
    """Check if .env.example exists and contains GROQ_API_KEY template."""
    env_example = PROJECT_ROOT / ".env.example"
    if not env_example.is_file():
        return False, ".env.example file not found"
    content = env_example.read_text(encoding="utf-8")
    if "GROQ_API_KEY=" in content:
        return True, ".env.example exists and contains GROQ_API_KEY template"
    return False, ".env.example exists but does not define GROQ_API_KEY"


def check_transformer_summary() -> tuple[bool, str]:
    """Check data/transformer_summary.txt exists and contains required factual terms."""
    summary_path = PROJECT_ROOT / "data" / "transformer_summary.txt"
    if not summary_path.is_file():
        return False, f"Missing file: {summary_path}"

    content = summary_path.read_text(encoding="utf-8")
    char_count = len(content)

    required_terms = [
        "28.4 BLEU for English-to-German translation",
        "41.8 BLEU for English-to-French translation",
        "six encoder layers and six decoder layers",
        "multi-head attention",
        "scaled dot-product attention",
        "adam optimizer",
        "positional encodings",
    ]

    missing = []
    content_lower = content.lower()
    for term in required_terms:
        if term.lower() not in content_lower:
            missing.append(term)

    if missing:
        return False, f"Missing required factual terms: {missing}"

    return True, (
        f"Corpus validated ({char_count} chars, ~1200 chars targeted). "
        "All 7 required factual statements are present."
    )


def check_project_structure() -> tuple[bool, str]:
    """Verify all required directories and placeholder files exist."""
    required_paths = [
        "app.py",
        "src/__init__.py",
        "src/state.py",
        "src/store.py",
        "src/nodes.py",
        "src/graph.py",
        "src/prompts.py",
        "data/transformer_summary.txt",
        "eval/__init__.py",
        "eval/run_eval.py",
        "tests/__init__.py",
        ".env",
        ".env.example",
        ".gitignore",
        "requirements.txt",
        "README.md",
    ]

    missing = []
    for rel_path in required_paths:
        p = PROJECT_ROOT / rel_path
        if not p.exists():
            missing.append(rel_path)

    if missing:
        return False, f"Missing required paths: {missing}"
    return True, f"All {len(required_paths)} required project files/directories exist"


def check_configuration_loader() -> tuple[bool, str]:
    """Verify config validation function works safely without revealing credentials."""
    try:
        from src.config import validate_groq_config, is_groq_configured, PLACEHOLDER_KEY

        # Test validation with current environment state
        status = validate_groq_config()
        if not isinstance(status, dict) or "is_valid" not in status or "message" not in status:
            return False, "validate_groq_config did not return expected dict structure"

        # Verify raw secret is never included in the message or return structure
        sample_secret = "mock_test_super_secret_1234567890abcdef"
        os.environ["GROQ_API_KEY"] = sample_secret
        test_status = validate_groq_config()
        if sample_secret in str(test_status):
            return False, "Raw API key was leaked in validation status!"
        if test_status.get("masked_key") == sample_secret:
            return False, "Raw API key was leaked in masked_key field!"

        # Reset environment back to empty / placeholder
        os.environ["GROQ_API_KEY"] = ""
        empty_status = validate_groq_config()
        if empty_status["is_valid"]:
            return False, "Empty key incorrectly validated as valid"

        os.environ["GROQ_API_KEY"] = PLACEHOLDER_KEY
        placeholder_status = validate_groq_config()
        if placeholder_status["is_valid"]:
            return False, "Placeholder key incorrectly validated as valid"

        # Restore original .env file state
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=True)

        return True, "Configuration loader securely validated without exposing credentials"
    except Exception as e:
        return False, f"Configuration loader check raised exception: {e}"


def run_all_checks() -> bool:
    """Run all Phase 1 checks and print PASS/FAIL status."""
    checks = [
        ("1. Python Version Compatibility", check_python_version),
        ("2. Required Dependencies Import", check_required_imports),
        ("3. Environment Template (.env.example)", check_env_example),
        ("4. Fixed Corpus (data/transformer_summary.txt)", check_transformer_summary),
        ("5. Project Directory Structure", check_project_structure),
        ("6. Secure Configuration Loader", check_configuration_loader),
    ]

    print("=" * 70)
    print("ALETHIA - PHASE 1 ENVIRONMENT & FOUNDATION VALIDATION")
    print("=" * 70)

    all_passed = True
    for title, check_fn in checks:
        passed, msg = check_fn()
        status_tag = "[PASS]" if passed else "[FAIL]"
        print(f"{status_tag} {title}")
        print(f"       Details: {msg}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("PHASE 1 VALIDATION RESULT: ALL CHECKS PASSED [OK]")
    else:
        print("PHASE 1 VALIDATION RESULT: CHECKS FAILED [ERROR]")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)

