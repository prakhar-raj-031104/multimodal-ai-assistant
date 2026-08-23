"""
Live multimodal demo dashboard (Gradio) — the HR/demo-friendly frontend.

Full browser experience, no server hardware required:
  * 📷 Camera   — uses the *browser's* webcam (works even if the server has none)
  * 🎤 Voice in — speak into the mic; transcribed with the same STT as the CLI
  * 💬 Chat     — streamed text answer, grounded in what it sees + remembers
  * 🔊 Voice out— the reply is spoken back in the page (gTTS)

    pip install gradio gtts
    python ui/gradio_app.py    # then open http://127.0.0.1:7860
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import config
from core.logging_setup import setup_logging, get_logger
from core.latency import stats_snapshot

log = get_logger("ui")


def main() -> None:
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio gtts")
        return

    setup_logging(config.log_level)
    from core.runtime import Assistant
    assistant = Assistant(config)

    def stats_md() -> str:
        s = stats_snapshot()
        lines = [
            f"**Brain:** `{config.llm.provider}` / "
            f"`{config.llm.gemini_model if config.llm.provider=='gemini' else config.llm.model}`",
            f"**Memory facts:** {len(assistant.memory.store)}",
            f"**Vision:** {'on' if assistant.vision.engine else 'off'}",
            f"**Voice out:** gTTS",
        ]
        for stage, v in s.items():
            lines.append(f"- `{stage}` p50={v['p50_ms']:.0f}ms p95={v['p95_ms']:.0f}ms")
        return "\n".join(lines)

    def _describe(image):
        """Run the VLM on a browser webcam frame (RGB numpy) -> scene summary."""
        if image is None or assistant.vision.engine is None:
            return None
        try:
            import cv2
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            raw = assistant.vision.engine.analyze_frame(bgr)
            return assistant.vision._to_perception(raw).summary
        except Exception as e:  # noqa
            log.debug("vision failed: %s", e)
            return None

    def respond(mic_path, image, typed, history):
        history = history or []
        # 1) user text — typed wins, else transcribe the mic clip
        user_text = (typed or "").strip()
        if not user_text and mic_path:
            user_text = assistant.stt.transcribe_file(mic_path)
        if not user_text:
            return history, None, "", None, stats_md()

        # 2) look at the camera frame (if the user shared one)
        scene = _describe(image)

        # 3) reason
        reply = "".join(assistant.brain.answer_stream(user_text, scene_summary=scene)).strip()
        if not reply:
            reply = "(no response)"

        # 4) speak the reply back in the browser
        audio = assistant.tts.synthesize_file(reply)

        history = history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ]
        return history, audio, "", None, stats_md()

    with gr.Blocks(title="Second Brain Assistant") as demo:
        gr.Markdown("# 🧠👓 Real-time Multimodal Second-Brain Assistant")
        gr.Markdown("Speak or type. Share your camera and ask *“what do you see?”* — "
                    "it answers in text **and** voice.")
        with gr.Row():
            with gr.Column(scale=1):
                cam = gr.Image(sources=["webcam"], type="numpy",
                               label="📷 Camera (your browser)", height=260)
                mic = gr.Audio(sources=["microphone"], type="filepath",
                               label="🎤 Speak, then Send")
                stats = gr.Markdown(stats_md())
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=380)
                with gr.Row():
                    txt = gr.Textbox(placeholder="…or type a message",
                                     show_label=False, scale=5, autofocus=True)
                    send = gr.Button("Send", variant="primary", scale=1)
                reply_audio = gr.Audio(label="🔊 Assistant voice", autoplay=True,
                                       interactive=False)

        outputs = [chatbot, reply_audio, txt, mic, stats]
        send.click(respond, [mic, cam, txt, chatbot], outputs)
        txt.submit(respond, [mic, cam, txt, chatbot], outputs)

    demo.launch()


if __name__ == "__main__":
    main()
