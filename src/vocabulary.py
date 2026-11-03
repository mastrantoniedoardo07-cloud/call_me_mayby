import json
from typing import Optional
from pydantic import BaseModel


class VocabularyError(Exception):
    pass


class Vocabulary(BaseModel):
    id_to_token: dict[int, str]
    size: int

    @classmethod
    def from_file(cls, path: str, size: int) -> "Vocabulary":
        """Load token mapping from JSON and build the ID-to-token dictionary."""
        # Load raw json data (text -> id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw: dict[str, int] = json.load(f)
        except FileNotFoundError:
            raise VocabularyError(f"File not found: {path}")
        except json.JSONDecodeError as e:
            raise VocabularyError(f"Invalid JSON in {path}: {e}")

        # Invert dictionary to map id -> text
        id_to_token = {token_id: text for text, token_id in raw.items()}
        
        # Check for data loss due to duplicate IDs
        if len(id_to_token) != len(raw):
            raise VocabularyError("data loss: duplicate IDs in vocabulary file")

        return cls(id_to_token=id_to_token, size=size)

    def get_text(self, token_id: int) -> Optional[str]:
        """Return the text corresponding to a token ID, handling special tokens."""
        # Ensure token ID is within valid range
        if token_id >= self.size:
            raise ValueError(f"Token ID {token_id} is out of bounds")
        
        # Return text if present, otherwise fallback for special tokens
        if token_id in self.id_to_token:
            return self.id_to_token[token_id]
            
        return f"<SPECIAL_TOKEN_{token_id}>"