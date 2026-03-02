import tensorflow as tf
from src.config import CFG, load_csv, make_label_mapping
from src.datasets import build_dataset
from src.model import build_video_vit_classifier, ViTConfig


def main():
    # test rapid: un batch din tf.data + forward pass
    train = load_csv(CFG.CSV_TRAIN, require_labels=True)
    num_classes, _, *_ = make_label_mapping(train)

    ds = build_dataset(train, split="Train", training=True, batch_size=2, debug=True)
    x, y = next(iter(ds))

    model = build_video_vit_classifier(
        num_frames=CFG.NUM_FRAMES,
        img_size=CFG.IMG_SIZE,
        num_classes=num_classes,
        vit_cfg=ViTConfig(img_size=CFG.IMG_SIZE)
    )

    out = model(x, training=False)
    print("x:", x.shape, x.dtype)
    print("y:", y.shape, y.dtype)
    print("out:", out.shape, out.dtype)
    print("sum probs sample0:", float(tf.reduce_sum(out[0])))

    model.summary()
    print("params:", model.count_params())


if __name__ == "__main__":
    main()