from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import gradio as gr

from inference import DEFAULT_TOP_K, VideoClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interfata de testare pentru modelul 3D CNN antrenat cu train_cnn.py."
    )
    parser.add_argument(
        "--model",
        default="../runs/cnn_3d/best_model_3d_cnn.keras",
        help="Calea catre modelul .keras.",
    )
    parser.add_argument(
        "--classes",
        default="../runs/cnn_3d/class_names.json",
        help="Calea catre class_names.json.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host pentru interfata web.",
    )
    parser.add_argument(
        "--port",
        default=7860,
        type=int,
        help="Port pentru interfata web.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Creeaza un link public Gradio. Optional, necesita internet.",
    )
    return parser.parse_args()


def make_top_predictions_table(labels: list[str], scores: list[float]) -> list[list[Any]]:
    return [[label, f"{score * 100:.2f}%"] for label, score in zip(labels, scores)]


def build_app(classifier: VideoClassifier) -> gr.Blocks:
    settings_text = (
        f"Model: {classifier.model_path.name} | "
        f"Cadre: {classifier.settings.num_frames} | "
        f"Imagine: {classifier.settings.img_size}x{classifier.settings.img_size} | "
        f"Clase: {len(classifier.class_names)}"
    )

    def stream_predict(frame, state):
        if state is None:
            state = {"frames": []}

        frames = deque(state.get("frames", []), maxlen=classifier.settings.num_frames)

        if frame is None:
            return "Camera nu trimite inca imagini.", [], {"frames": list(frames)}

        processed = classifier.preprocess_frame(frame)
        frames.append(processed)
        state = {"frames": list(frames)}

        if len(frames) < classifier.settings.num_frames:
            remaining = classifier.settings.num_frames - len(frames)
            return f"Se strang cadre pentru primul clip: mai lipsesc {remaining}.", [], state

        prediction = classifier.predict_clip(frames, top_k=DEFAULT_TOP_K)
        label = f"{prediction.label} ({prediction.confidence * 100:.2f}%)"
        table = make_top_predictions_table(prediction.top_labels, prediction.top_scores)
        return label, table, state

    with gr.Blocks(title="Testare model 3D CNN") as demo:
        gr.Markdown("# Testare model 3D CNN")
        gr.Markdown(settings_text)

        with gr.Row():
            webcam = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                label="Camera",
                height=420,
            )
            with gr.Column():
                predicted_label = gr.Label(label="Predictie curenta")
                top_predictions = gr.Dataframe(
                    headers=["Label", "Scor"],
                    datatype=["str", "str"],
                    label="Top predictii",
                    row_count=(DEFAULT_TOP_K, "fixed"),
                    col_count=(2, "fixed"),
                    interactive=False,
                )

        state = gr.State({"frames": []})
        webcam.stream(
            stream_predict,
            inputs=[webcam, state],
            outputs=[predicted_label, top_predictions, state],
            stream_every=0.25,
        )

    return demo


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    class_names_path = Path(args.classes)
    classifier = VideoClassifier(model_path, class_names_path)
    demo = build_app(classifier)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
