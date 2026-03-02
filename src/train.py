import json
from pathlib import Path

import os
import warnings

#pt un raspuns fara warning uri neimportante
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "0"
warnings.filterwarnings("ignore")


import tensorflow as tf
tf.get_logger().setLevel("ERROR")


from src.config import CFG, set_seed, load_csv, make_label_mapping, check_split
from src.datasets import build_dataset
from src.model import build_video_vit_classifier, ViTConfig

"""
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional, reduce variabilitatea
"""

# sanity run: pune True ca sa testezi rapid fit
SANITY_ONLY = False
SANITY_TRAIN_STEPS = 2
SANITY_VAL_STEPS = 1
SANITY_BATCH_SIZE = 2


EPOCHS = 30


def set_gpu_memory_growth() -> None:
    #nu lasa tf sa ia tot vram-ul
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[INFO] No GPU detected by TensorFlow.")
        return

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Enabled memory growth for {len(gpus)} GPU(s).")
    except Exception as e:
        print("[WARN] Could not set memory growth:", e)


def make_callbacks(out_dir: Path):
    # callbacks simple: checkpoint, early stop, lr reduce, logs
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "best_model_vit.keras"
    log_dir = out_dir / "logs"
    backup_dir = out_dir / "backup"

    return [
        # backup ca sa pot relua trainingul daca se inchide terminalul
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(backup_dir=str(backup_dir))
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_sparse_categorical_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=0,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(out_dir / "history.csv"),
            append=False,
        ),
    ]


def main():
    # 0) setup
    set_seed(CFG.SEED)
    set_gpu_memory_growth()

    """
    tf.config.optimizer.set_jit(False)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    """

    # 1) load csv
    train_df = load_csv(CFG.CSV_TRAIN, require_labels=True)
    val_df = load_csv(CFG.CSV_VAL, require_labels=True)

    # 2) labels
    num_classes, class_names, label_to_id, _ = make_label_mapping(train_df)
    check_split(val_df, label_to_id, "Validation")

    out_dir = Path("runs") / "vit_framewise"
    out_dir.mkdir(parents=True, exist_ok=True)

    # salveaza numele claselor
    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)


    # 3) datasets
    if SANITY_ONLY:
        batch_size = SANITY_BATCH_SIZE
        debug_data = True
    else:
        batch_size = CFG.BATCH_SIZE
        debug_data = False

    train_ds = build_dataset(train_df, split="Train", training=True, batch_size=batch_size, debug=debug_data)
    val_ds = build_dataset(val_df, split="Validation", training=False, batch_size=batch_size, debug=debug_data)

    # 4) model
    vit_cfg = ViTConfig(
        img_size=CFG.IMG_SIZE,
        patch_size=32,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1,
        )

    model = build_video_vit_classifier(
        num_frames=CFG.NUM_FRAMES,
        img_size=CFG.IMG_SIZE,
        num_classes=num_classes,
        vit_cfg=vit_cfg,
        head_hidden=256,
        head_dropout=0.2,
        )

    # 5) compile
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)
        opt_name = "AdamW"
        weight_decay = 1e-4
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        opt_name = "Adam"
        weight_decay = None
        print("[warn] adamw not available, fallback to adam")

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )

    model.summary()

    # sanity: forward pass pe un batch
    x0, y0 = next(iter(train_ds))
    yhat0 = model(x0, training=False)
    print("[sanity] x:", x0.shape, x0.dtype)
    print("[sanity] y:", y0.shape, y0.dtype)
    print("[sanity] yhat:", yhat0.shape, "sum probs[0]:", float(tf.reduce_sum(yhat0[0])))

    # salveaza config-ul rularii
    run_cfg = {
        "seed": CFG.SEED,
        "batch_size": batch_size,
        "num_frames": CFG.NUM_FRAMES,
        "img_size": CFG.IMG_SIZE,
        "num_classes": num_classes,
        "vit_cfg": vit_cfg.__dict__,
        "head_hidden": 256,
        "head_dropout": 0.2,
        "optimizer": opt_name,
        "lr": 3e-4,
        "weight_decay": weight_decay,
        "epochs": EPOCHS,
        "sanity_only": SANITY_ONLY,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    # 6) train
    callbacks = make_callbacks(out_dir)

    if SANITY_ONLY:
        print(f"[sanity] mini-fit: train.take({SANITY_TRAIN_STEPS}), val.take({SANITY_VAL_STEPS})")
        model.fit(
            train_ds.take(SANITY_TRAIN_STEPS),
            validation_data=val_ds.take(SANITY_VAL_STEPS),
            epochs=1,
            callbacks=callbacks,
        )
        print("[sanity] ok. pune SANITY_ONLY=False pentru training complet")
        return

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # 7) save last model
    model.save(out_dir / "last_model_vit.keras")
    print(f"saved best model to: {out_dir / 'best_model_vit.keras'}")
    print(f"saved last model to: {out_dir / 'last_model_vit.keras'}")
    print(f"saved class names to: {out_dir / 'class_names.json'}")
    print(f"saved run config to: {out_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()