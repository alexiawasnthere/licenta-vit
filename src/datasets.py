from typing import Literal
import pandas as pd
import tensorflow as tf

from src.config import CFG
from src.data_utils import load_clip_and_label_tf

SplitName = Literal["Train", "Validation", "Test"]


def build_dataset(
    df: pd.DataFrame,
    split,
    training: bool,
    batch_size: int = CFG.BATCH_SIZE,
    debug: bool = False,
    shuffle_buffer: int = 2048
    ) -> tf.data.Dataset:
    # creeaza dataset (video_id, label_id) -> (clip, label_id)
    video_ids = df[CFG.COL_VIDEO_ID].to_numpy("int32")
    label_ids = df[CFG.COL_LABEL_ID].to_numpy("int32")

    ds = tf.data.Dataset.from_tensor_slices((video_ids, label_ids))

    if training:
        ds = ds.shuffle(
            buffer_size=min(shuffle_buffer, len(video_ids)),
            seed=CFG.SEED,
            reshuffle_each_iteration=True
            )

    # optiuni tf.data (debug = deterministic si mai lent)
    options = tf.data.Options()
    if debug:
        options.experimental_deterministic = True
    else:
        options.experimental_deterministic = False
    ds = ds.with_options(options)

    data_root = str(CFG.DATA_ROOT)
    num_frames = CFG.NUM_FRAMES
    img_size = CFG.IMG_SIZE

    ds = ds.map(
        lambda vid, lid: load_clip_and_label_tf(
            data_root, 
            num_frames, 
            img_size, 
            split, 
            vid, 
            lid, 
            training
        ),
        num_parallel_calls=(1 if debug else tf.data.AUTOTUNE)
        )

    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(1 if debug else tf.data.AUTOTUNE)

    return ds

"""
    def build_all_datasets(train_df: pd.DataFrame, 
                       val_df: pd.DataFrame, 
                       test_df: pd.DataFrame
                       ):
    train_ds = build_dataset(train_df, split="Train", training=True)
    val_ds = build_dataset(val_df, split="Validation", training=False)
    #test_ds = build_dataset(test_df, split="Test", training=False)
    return train_ds, val_ds#, test_ds
"""