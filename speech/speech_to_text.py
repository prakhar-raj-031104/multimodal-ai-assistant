"""Audio transcription utilities using Voxtral Mini 4B Realtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


AudioPath = Union[str, Path]


@dataclass
class AudioTranscriptionResult:
    """Structured result returned by the audio transcription module."""

    text: str
    model_id: str


class VoxtralSpeechToText:
    """Thin wrapper around Voxtral Mini 4B Realtime for transcription."""

    DEFAULT_MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        try:
            from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Voxtral dependencies are missing. Install `transformers>=5.2.0`, "
                "`mistral-common[audio]`, and `torch` to enable audio transcription."
            ) from exc

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
        )

    def transcribe(self, audio_path: AudioPath, prompt: Optional[str] = None) -> AudioTranscriptionResult:
        """
        Transcribe an audio file into text.

        Args:
            audio_path: Path to a local audio file.
            prompt: Optional transcription instruction.

        Returns:
            AudioTranscriptionResult with extracted text.
        """

        self._lazy_load()

        from mistral_common.tokens.tokenizers.audio import Audio

        instruction = prompt or "Transcribe this audio as accurately as possible."
        normalized_path = str(Path(audio_path).expanduser().resolve())

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": Audio(url=normalized_path),
                    },
                    {
                        "type": "text",
                        "text": instruction,
                    },
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            conversation,
            return_tensors="pt",
            truncation=True,
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
        )

        text = self._processor.batch_decode(
            outputs,
            skip_special_tokens=True,
        )[0].strip()

        return AudioTranscriptionResult(text=text, model_id=self.model_id)


def transcribe_audio_file(audio_path: AudioPath, prompt: Optional[str] = None) -> str:
    """Convenience helper used by the assistant pipeline."""

    return VoxtralSpeechToText().transcribe(audio_path=audio_path, prompt=prompt).text
