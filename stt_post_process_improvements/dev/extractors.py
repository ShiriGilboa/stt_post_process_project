from abc import ABC, abstractmethod
import json
import re
from typing import Any, Optional


class BaseResponseExtractor(ABC):
    """
    Abstract base class for response extraction strategies.
    """
    @abstractmethod
    def extract(self, response: str) -> Any:
        """
        Extract structured data from the raw LLM response string.

        Args:
            response: Raw text output from the LLM.

        Returns:
            Parsed/processed data in a structured form.
        """
        pass

class JsonResponseExtractor(BaseResponseExtractor):
    """
    Extractor that parses the LLM response as JSON and optionally retrieves a specific key.
    """
    def __init__(self, key: Optional[str] = None, def_val: str = ""):
        """
        Args:
            key: If provided, extract the corresponding value from the parsed JSON.
        """
        self.key = key
        self.def_val = def_val

    def _clean_response(self, response: str) -> str:
        # 1) Remove markdown fences and language hints
        cleaned = re.sub(r"```(?:json)?", "", response, flags=re.IGNORECASE)
        # 2) Trim stray backticks
        cleaned = cleaned.strip("`\n ")
        # 3) Extract only the first {...} valid JSON object
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return match.group(0)
        return cleaned

    def extract(self, response: str) -> Any:
        cleaned = self._clean_response(response)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON:\n{e}\n\nCleaned response was:\n{cleaned!r}")

        if self.key is None:
            return data

        # support nested keys via dot‑notation
        current = data
        for part in self.key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise KeyError(f"Key '{self.key}' not found in JSON")
        return current