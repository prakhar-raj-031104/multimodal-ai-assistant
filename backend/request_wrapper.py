from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class MultimodalRequest:
    user_input: Optional[str] = None
    vision_data: Optional[Dict] = None
    audio_data: Optional[str] = None
    memory_data: Optional[str] = None