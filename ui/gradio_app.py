"""
Optional live dashboard (Gradio).

A lightweight way to talk to the assistant and watch its state — streaming chat,
current scene, and memory/latency stats — without wiring up mic/camera.

    pip install gradio
    python ui/gradio_app.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import config
from core.logging_setup import setup_logging
from core.latency import stats_snapshot


def main() -> None:
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        return

    setup_logging(config.log_level)
    from core.runtime import Assistant
    assistant = Assistant(config)

    def stats_md() -> str:
        s = stats_snapshot()
        lines = [
            f"**Model:** `{config.llm.model}`",
            f"**Memory facts:** {len(assistant.memory.store)}",
            f"**Episodic events:** {len(assistant.memory.episodic)}",
            f"**Vision:** {'on' if assistant.vision.engine else 'off'}",
        ]
        for stage, v in s.items():
            lines.append(f"- `{stage}` p50={v['p50_ms']:.0f}ms p95={v['p95_ms']:.0f}ms")
        return "\n".join(lines)

    def respond(message, history):
        """Stream the assistant reply token-by-token into the chat."""
        history = (history or []) + [{"role": "user", "content": message}]
        history.append({"role": "assistant", "content": ""})
        scene = assistant._current_scene()
        for chunk in assistant.brain.answer_stream(message, scene_summary=scene):
            history[-1]["content"] += chunk
            yield "", history, stats_md()

    with gr.Blocks(title="Second Brain Assistant") as demo:
        gr.Markdown("# 🧠👓 Real-time Multimodal Second-Brain Assistant")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=460, label="Chat")
                with gr.Row():
                    msg = gr.Textbox(placeholder="Ask me anything…", scale=5,
                                     show_label=False, autofocus=True)
                    send = gr.Button("Send", variant="primary", scale=1)
            with gr.Column(scale=1):
                stats = gr.Markdown(stats_md())
                gr.Button("Refresh").click(stats_md, outputs=stats)

        send.click(respond, [msg, chatbot], [msg, chatbot, stats])
        msg.submit(respond, [msg, chatbot], [msg, chatbot, stats])

    demo.launch()


if __name__ == "__main__":
    main()
