import tensorflow as tf
from src.config import CFG, load_csv
from src.data_utils import load_clip


def main():
    # test rapid: un clip incarcat in eager
    train = load_csv(CFG.CSV_TRAIN, require_labels=True)

    ex_id = int(train.iloc[0][CFG.COL_VIDEO_ID])
    clip = load_clip(CFG.DATA_ROOT, CFG.NUM_FRAMES, CFG.IMG_HEIGHT, CFG.IMG_WIDTH, ex_id, "Train")

    print("video_id:", ex_id)
    print("clip shape:", clip.shape)
    print("dtype:", clip.dtype)
    print("min/max:", float(tf.reduce_min(clip)), float(tf.reduce_max(clip)))


if __name__ == "__main__":
    main()