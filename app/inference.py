from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf


DEFAULT_NUM_FRAMES = 37
DEFAULT_IMG_SIZE = 128
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class ModelSettings:
    num_frames: int = DEFAULT_NUM_FRAMES
    img_size: int = DEFAULT_IMG_SIZE


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    top_labels: list[str]
    top_scores: list[float]


class VideoClassifier:
    def __init__(
        self,
        model_path: str | Path,
        class_names_path: str | Path,
        settings: ModelSettings | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.class_names_path = Path(class_names_path).expanduser().resolve()
        self.settings = settings or load_settings_near_model(self.model_path)
        self.class_names = load_class_names(self.class_names_path)
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def preprocess_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        if frame_rgb is None:
            raise ValueError("Frame gol primit de la camera.")

        frame = np.asarray(frame_rgb)
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise ValueError(f"Frame invalid: shape={frame.shape}")

        if frame.shape[-1] == 4:
            frame = frame[..., :3]

        frame = cv2.resize(
            frame,
            (self.settings.img_size, self.settings.img_size),
            interpolation=cv2.INTER_AREA,
        )
        return frame.astype(np.float32) / 255.0

    def predict_clip(
        self,
        frames: Iterable[np.ndarray],
        top_k: int = DEFAULT_TOP_K,
    ) -> Prediction:
        clip = np.stack(list(frames), axis=0).astype(np.float32)
        expected_shape = (
            self.settings.num_frames,
            self.settings.img_size,
            self.settings.img_size,
            3,
        )
        if clip.shape != expected_shape:
            raise ValueError(f"Clip invalid: shape={clip.shape}, asteptat={expected_shape}")

        probs = self.model.predict(clip[None, ...], verbose=0)[0]
        top_k = max(1, min(top_k, len(probs), len(self.class_names)))
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_labels = [self.class_names[int(i)] for i in top_indices]
        top_scores = [float(probs[int(i)]) for i in top_indices]

        return Prediction(
            label=top_labels[0],
            confidence=top_scores[0],
            top_labels=top_labels,
            top_scores=top_scores,
        )


def load_class_names(path: str | Path) -> list[str]:
    class_names_path = Path(path).expanduser().resolve()
    if not class_names_path.exists():
        raise FileNotFoundError(f"Nu exista fisierul de clase: {class_names_path}")

    with class_names_path.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    if not isinstance(class_names, list) or not all(isinstance(x, str) for x in class_names):
        raise ValueError("class_names.json trebuie sa contina o lista de string-uri.")

    return class_names


def load_settings_near_model(model_path: str | Path) -> ModelSettings:
    model_path = Path(model_path).expanduser().resolve()
    run_config_path = model_path.parent / "run_config.json"
    if not run_config_path.exists():
        return ModelSettings()

    with run_config_path.open("r", encoding="utf-8") as f:
        run_config = json.load(f)

    return ModelSettings(
        num_frames=int(run_config.get("num_frames", DEFAULT_NUM_FRAMES)),
        img_size=int(run_config.get("img_size", DEFAULT_IMG_SIZE)),
    )
