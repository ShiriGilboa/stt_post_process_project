import re
import yaml
import num2words
from typing import List, Tuple


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()

    def replace_number(match):
        try:
            return num2words.num2words(int(match.group()))
        except Exception:
            return match.group()

    # Replace numbers with words **before** removing special characters
    text = re.sub(r'\d+', replace_number, text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_yaml_block(text: str) -> Tuple[str, List[str]]:
    """
    Extract the fenced ```yaml ... ``` block and return
    (short_context_description, list_of_terms).

    If no block or keys are missing, returns ("", []).
    """
    pattern = r"```yaml\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return "", []

    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        print(f"[WARN] YAML parse error: {exc}")
        return "", []

    desc = (data.get("short_context_description") or "").strip()

    terms = (
        data.get("nlist_of_terms")
        or data.get("list_of_terms")
        or data.get("list_of_unique_terms")
        or []
    )
    # Allow comma-separated string as fallback
    if isinstance(terms, str):
        terms = [s.strip() for s in re.split(r",\s*", terms) if s.strip()]

    return desc, terms


def build_plain_prompt(desc: str, terms: List[str]) -> str:
    """
    Convert (description, glossary) into Whisper-friendly plain text.
    """
    glossary = ", ".join(terms)
    if desc and glossary:
        return f"{desc}. It's {glossary}."
    return desc or glossary
