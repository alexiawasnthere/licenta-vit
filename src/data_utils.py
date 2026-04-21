from pathlib import Path
from typing import Literal, Tuple

import tensorflow as tf

from src.data_paths import get_frame_paths, make_frame_path_tf

SplitName = Literal["Train", "Validation", "Test"]

def decode_and_resize_jpeg(
    image_bytes: tf.Tensor,
    img_height: int,
    img_width: int,
    training: bool
) -> tf.Tensor:
    img = tf.image.decode_jpeg(image_bytes, channels=3)   # uint8
    img = tf.image.convert_image_dtype(img, tf.float32)   # float32 in [0, 1]
    img = tf.image.resize(img, (img_height, img_width))   # (H, W)

    if training: #augmentare doar pe training
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)

    return img


def load_frame(
    path: tf.Tensor,
    img_height: int,
    img_width: int,
    training: bool
) -> tf.Tensor:
    # citeste un jpeg de pe disc si il preproceseaza
    image_bytes = tf.io.read_file(path)
    return decode_and_resize_jpeg(image_bytes, img_height, img_width, training)


def load_clip(
    data_root: Path,
    num_frames: int,
    img_height: int,
    img_width: int,
    video_id: int,
    split: SplitName
) -> tf.Tensor:
    # varianta python (pentru test / debug)
    frame_paths = get_frame_paths(data_root, num_frames, video_id, split, strict=True)
    frame_strs = [str(p) for p in frame_paths]
    frames = [load_frame(tf.constant(p), img_height, img_width, training= False) for p in frame_strs]
    return tf.stack(frames, axis=0)  # (t, h, w, c)


def load_clip_and_label(
    data_root: Path,
    num_frames: int,
    img_height: int,
    img_width: int,
    video_id: int,
    label_id: int,
    split: SplitName
) -> Tuple[tf.Tensor, tf.Tensor]:
    # (clip, label_id) pentru debug
    clip = load_clip(data_root, num_frames, img_height, img_width, video_id, split)
    y = tf.convert_to_tensor(label_id, dtype=tf.int32)
    return clip, y


def load_clip_tf(
    data_root: str,
    num_frames: int,
    img_height: int,
    img_width: int,
    split: str,
    video_id: tf.Tensor,
    training: bool
) -> tf.Tensor:
    # varianta tf.data friendly (foloseste tf strings)
    data_root_t = tf.constant(data_root)
    split_t = tf.constant(split)

    frame_indices = tf.range(1, num_frames + 1, dtype=tf.int32)  # 1..T

    paths = tf.map_fn(
        lambda i: make_frame_path_tf(data_root_t, split_t, tf.cast(video_id, tf.int32), i),
        frame_indices,
        fn_output_signature=tf.string
    )

    frames = tf.map_fn(
        lambda p: load_frame(p, img_height, img_width, False),  # fără augment aici
        paths,
        fn_output_signature=tf.float32
)
    if training:
        flip = tf.random.uniform([]) > 0.5

        if flip:
            frames = tf.image.flip_left_right(frames)

        frames = tf.image.random_brightness(frames, 0.1)
        frames = tf.image.random_contrast(frames, 0.8, 1.2)

    # ajuta keras la shape inference
    frames.set_shape([num_frames, img_height, img_width, 3])
    return frames


def load_clip_and_label_tf(
    data_root: str,
    num_frames: int,
    img_height: int,
    img_width: int,
    split: str,
    video_id: tf.Tensor,
    label_id: tf.Tensor,
    training: bool
) -> Tuple[tf.Tensor, tf.Tensor]:
    clip = load_clip_tf(data_root, num_frames, img_height, img_width, split, video_id, training)
    label_id = tf.cast(label_id, tf.int32)
    return clip, label_id