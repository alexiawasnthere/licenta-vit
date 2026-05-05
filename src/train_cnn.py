import json
import os
import warnings
from pathlib import Path

# pt un raspuns fara warning-uri neimportante
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION"] = "0"
warnings.filterwarnings("ignore")

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from src.config import CFG, check_split, load_csv, make_label_mapping, set_seed
from src.datasets import build_dataset


# sanity run: pune True ca sa testezi rapid fit
SANITY_ONLY = False
SANITY_TRAIN_STEPS = 2
SANITY_VAL_STEPS = 1
SANITY_BATCH_SIZE = 2

EPOCHS = 30
TOP_K_CLASSES = 27


def set_gpu_memory_growth() -> None:
    # nu lasa tf sa ia tot vram-ul
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


def conv3d_block(
    x: tf.Tensor,
    filters: int,
    pool_size: tuple[int, int, int],
    name: str,
    dropout: float = 0.0,
) -> tf.Tensor:
    x = tf.keras.layers.Conv3D(
        filters,
        kernel_size=(3, 3, 3),
        padding="same",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=f"{name}_conv1",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name}_relu1")(x)

    x = tf.keras.layers.Conv3D(
        filters,
        kernel_size=(3, 3, 3),
        padding="same",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=f"{name}_conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name}_relu2")(x)

    x = tf.keras.layers.MaxPooling3D(pool_size=pool_size, name=f"{name}_pool")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")(x)
    return x


def build_video_3d_cnn_classifier(
    num_frames: int,
    img_size: int,
    num_classes: int,
    base_filters: int = 16,
    head_hidden: int = 256,
    head_dropout: float = 0.4,
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(
        shape=(num_frames, img_size, img_size, 3),
        name="clip",
    )

    x = conv3d_block(
        inputs,
        filters=base_filters,
        pool_size=(1, 2, 2),
        dropout=0.1,
        name="block1",
    )
    x = conv3d_block(
        x,
        filters=base_filters * 2,
        pool_size=(2, 2, 2),
        dropout=0.15,
        name="block2",
    )
    x = conv3d_block(
        x,
        filters=base_filters * 4,
        pool_size=(2, 2, 2),
        dropout=0.2,
        name="block3",
    )
    x = conv3d_block(
        x,
        filters=base_filters * 8,
        pool_size=(2, 2, 2),
        dropout=0.25,
        name="block4",
    )

    x = tf.keras.layers.GlobalAveragePooling3D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(
        head_hidden,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="head_dense",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(head_dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)

    return tf.keras.Model(inputs, outputs, name="video_3d_cnn")


def make_callbacks(out_dir: Path):
    # callbacks simple: checkpoint, lr reduce, logs
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "best_model_3d_cnn.keras"
    log_dir = out_dir / "logs"
    backup_dir = out_dir / "backup"

    return [
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(backup_dir)),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_sparse_categorical_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=0),
        tf.keras.callbacks.CSVLogger(
            filename=str(out_dir / "history.csv"),
            append=False,
        ),
    ]


def filter_top_classes(train_df, val_df):
    top_labels = (
        train_df[CFG.COL_LABEL]
        .value_counts()
        .head(TOP_K_CLASSES)
        .index
        .tolist()
    )

    train_df = train_df[train_df[CFG.COL_LABEL].isin(top_labels)].copy()
    val_df = val_df[val_df[CFG.COL_LABEL].isin(top_labels)].copy()

    # refacem label_id-urile ca sa fie consecutive, incepand de la 0
    label_to_new_id = {label: i for i, label in enumerate(top_labels)}
    train_df[CFG.COL_LABEL_ID] = train_df[CFG.COL_LABEL].map(label_to_new_id).astype(int)
    val_df[CFG.COL_LABEL_ID] = val_df[CFG.COL_LABEL].map(label_to_new_id).astype(int)

    print("Clase folosite:", top_labels)
    print("Train rows dupa filtrare:", len(train_df))
    print("Val rows dupa filtrare:", len(val_df))

    return train_df, val_df, top_labels


def main():
    # 0) setup
    set_seed(CFG.SEED)
    set_gpu_memory_growth()
    tf.config.optimizer.set_jit(False)

    # 1) load csv
    train_df = load_csv(CFG.CSV_TRAIN, require_labels=True)
    val_df = load_csv(CFG.CSV_VAL, require_labels=True)
    train_df, val_df, top_labels = filter_top_classes(train_df, val_df)

    # 2) labels
    num_classes, class_names, label_to_id, _ = make_label_mapping(train_df)
    check_split(val_df, label_to_id, "Validation")

    out_dir = Path("runs") / "cnn_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # 3) datasets
    if SANITY_ONLY:
        batch_size = SANITY_BATCH_SIZE
        debug_data = True
    else:
        batch_size = CFG.BATCH_SIZE
        debug_data = False

    train_ds = build_dataset(
        train_df,
        split="Train",
        training=True,
        batch_size=batch_size,
        debug=debug_data,
    )
    val_ds = build_dataset(
        val_df,
        split="Validation",
        training=False,
        batch_size=batch_size,
        debug=debug_data,
    )

    # 4) model
    model = build_video_3d_cnn_classifier(
        num_frames=CFG.NUM_FRAMES,
        img_size=CFG.IMG_SIZE,
        num_classes=num_classes,
        base_filters=16,
        head_hidden=256,
        head_dropout=0.4,
    )

    # 5) compile
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=3e-4,
            weight_decay=1e-4,
            clipnorm=1.0,
        )
        opt_name = "AdamW"
        weight_decay = 1e-4
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=3e-4,
            clipnorm=1.0,
        )
        opt_name = "Adam"
        weight_decay = None
        print("[warn] adamw not available, fallback to adam")

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="sparse_categorical_accuracy"
            ),
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

    run_cfg = {
        "seed": CFG.SEED,
        "batch_size": batch_size,
        "num_frames": CFG.NUM_FRAMES,
        "img_size": CFG.IMG_SIZE,
        "num_classes": num_classes,
        "top_k_classes": TOP_K_CLASSES,
        "top_labels": top_labels,
        "model": "video_3d_cnn",
        "base_filters": 16,
        "head_hidden": 256,
        "head_dropout": 0.4,
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
        print(
            f"[sanity] mini-fit: train.take({SANITY_TRAIN_STEPS}), "
            f"val.take({SANITY_VAL_STEPS})"
        )
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
    model.save(out_dir / "last_model_3d_cnn.keras")
    print(f"saved best model to: {out_dir / 'best_model_3d_cnn.keras'}")
    print(f"saved last model to: {out_dir / 'last_model_3d_cnn.keras'}")
    print(f"saved class names to: {out_dir / 'class_names.json'}")
    print(f"saved run config to: {out_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
