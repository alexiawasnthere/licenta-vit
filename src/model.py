from dataclasses import dataclass
import tensorflow as tf
layers = tf.keras.layers


@dataclass(frozen=True)
class ViTConfig:
    img_size: int = 128
    patch_size: int = 8
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.1


class AddCLSToken(layers.Layer):
    def __init__(self, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.cls = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        # x: (B, N, D)
        b = tf.shape(x)[0]
        cls_tokens = tf.tile(self.cls, [b, 1, 1])   # (B, 1, D)
        return tf.concat([cls_tokens, x], axis=1)   # (B, N+1, D)



class AddPositionEmbedding(layers.Layer):
    def __init__(self, num_tokens: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos = self.add_weight(
            name="pos_embedding",
            shape=(1, self.num_tokens, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, x):
        return x + self.pos


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )
        self.drop1 = layers.Dropout(dropout)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp1 = layers.Dense(mlp_dim, activation=tf.nn.gelu)
        self.drop2 = layers.Dropout(dropout)
        self.mlp2 = layers.Dense(embed_dim)
        self.drop3 = layers.Dropout(dropout)

    def call(self, x, training=False):
        y = self.norm1(x)
        y = self.attn(y, y, training=training)
        y = self.drop1(y, training=training)
        x = x + y

        # MLP + residual
        y = self.norm2(x)
        y = self.mlp1(y)
        y = self.drop2(y, training=training)
        y = self.mlp2(y)
        y = self.drop3(y, training=training)
        return x + y


class TemporalAttentionPooling(layers.Layer):
    """
    Primeste secventa (B, T, D) si invata ponderi de atentie pe timp.
    Intoarce un singur vector (B, D).
    """
    def __init__(self, attn_hidden: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.score_mlp = tf.keras.Sequential([
            layers.Dense(attn_hidden, activation="tanh"),
            layers.Dense(1)
        ])

    def call(self, x):
        # x: (B, T, D)
        scores = self.score_mlp(x)                 # (B, T, 1)
        weights = tf.nn.softmax(scores, axis=1)   # (B, T, 1)
        context = tf.reduce_sum(weights * x, axis=1)  # (B, D)
        return context


def build_frame_cnn_encoder(
    img_size: int,
    backbone: str = "efficientnet_b0",
    weights: str | None = "imagenet",
    trainable_backbone: bool = False,
    embedding_dim: int = 256,
    dropout: float = 0.2,
) -> tf.keras.Model:
    inputs = layers.Input(shape=(img_size, img_size, 3), name="frame")

    # Dataset-ul este incarcat in [0, 1]. Modelele Keras pretrained asteapta
    # valori de tip imagine in [0, 255] si isi aplica preprocessing-ul intern.
    x = layers.Rescaling(255.0, name="to_255")(inputs)

    if backbone == "efficientnet_b0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=(img_size, img_size, 3),
            pooling="avg",
        )
    elif backbone == "mobilenet_v3_small":
        base = tf.keras.applications.MobileNetV3Small(
            include_top=False,
            weights=weights,
            input_shape=(img_size, img_size, 3),
            pooling="avg",
        )
    else:
        raise ValueError(
            "backbone trebuie sa fie 'efficientnet_b0' sau 'mobilenet_v3_small'"
        )

    base.trainable = trainable_backbone
    x = base(x, training=trainable_backbone)
    x = layers.Dropout(dropout, name="frame_dropout")(x)
    x = layers.Dense(embedding_dim, activation=tf.nn.gelu, name="frame_embedding")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="frame_norm")(x)

    return tf.keras.Model(inputs, x, name=f"{backbone}_frame_encoder")


def build_vit_encoder(cfg: ViTConfig) -> tf.keras.Model:
    if cfg.img_size % cfg.patch_size != 0:
        raise ValueError("img_size trebuie sa fie divizibil cu patch_size")

    num_patches_per_side = cfg.img_size // cfg.patch_size
    num_patches = num_patches_per_side * num_patches_per_side
    num_tokens = num_patches + 1  # + CLS

    inputs = layers.Input(shape=(cfg.img_size, cfg.img_size, 3), name="frame")

    # patch embedding
    x = layers.Conv2D(
        filters=cfg.embed_dim,
        kernel_size=cfg.patch_size,
        strides=cfg.patch_size,
        padding="valid",
        name="patch_embed"
    )(inputs)

    x = layers.Reshape((num_patches, cfg.embed_dim), name="flatten_patches")(x)

    # CLS + positional embedding
    x = AddCLSToken(cfg.embed_dim, name="add_cls")(x)
    x = AddPositionEmbedding(num_tokens, cfg.embed_dim, name="add_pos")(x)
    x = layers.Dropout(cfg.dropout, name="token_dropout")(x)

    # transformer encoder
    for i in range(cfg.depth):
        x = TransformerBlock(
            cfg.embed_dim,
            cfg.num_heads,
            cfg.mlp_dim,
            cfg.dropout,
            name=f"block_{i}"
        )(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")(x)

    # CLS token = reprezentarea finala a frame-ului
    x = layers.Lambda(lambda t: t[:, 0, :], name="cls_select")(x)

    return tf.keras.Model(inputs, x, name="vit_encoder")


def build_video_cnn_lstm_classifier(
    num_frames: int,
    img_size: int,
    num_classes: int,
    backbone: str = "efficientnet_b0",
    weights: str | None = "imagenet",
    trainable_backbone: bool = False,
    embedding_dim: int = 256,
    lstm_units: int = 256,
    head_hidden: int = 256,
    head_dropout: float = 0.2,
) -> tf.keras.Model:
    frame_encoder = build_frame_cnn_encoder(
        img_size=img_size,
        backbone=backbone,
        weights=weights,
        trainable_backbone=trainable_backbone,
        embedding_dim=embedding_dim,
        dropout=head_dropout,
    )

    inputs = layers.Input(shape=(num_frames, img_size, img_size, 3), name="clip")

    x = layers.TimeDistributed(frame_encoder, name="cnn_per_frame")(inputs)
    x = layers.LayerNormalization(epsilon=1e-6, name="temporal_norm")(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),
        name="temporal_bilstm",
    )(x)
    x = layers.Dropout(head_dropout, name="temporal_dropout")(x)

    x = TemporalAttentionPooling(attn_hidden=head_hidden, name="temporal_attention")(x)

    x = layers.Dense(head_hidden, activation=tf.nn.gelu, name="head_dense")(x)
    x = layers.Dropout(head_dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)

    return tf.keras.Model(inputs, outputs, name=f"video_{backbone}_bilstm_attn")


def build_video_vit_classifier(
    num_frames: int,
    img_size: int,
    num_classes: int,
    vit_cfg: ViTConfig | None = None,
    head_hidden: int = 256,
    head_dropout: float = 0.2,
) -> tf.keras.Model:
    vit_cfg = vit_cfg or ViTConfig(img_size=img_size)
    vit_encoder = build_vit_encoder(vit_cfg)

    inputs = layers.Input(shape=(num_frames, img_size, img_size, 3), name="clip")

    # 1) embedding per frame
    x = layers.TimeDistributed(vit_encoder, name="vit_per_frame")(inputs)   # (B, T, 256)

    # 2) normalizare temporala
    x = layers.LayerNormalization(epsilon=1e-6, name="temporal_norm")(x)

    # 3) modelare temporala
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True),
    )(x)
    x = layers.Dropout(0.2, name="temporal_dropout")(x)

    # 4) attention pooling pe timp
    x = TemporalAttentionPooling(attn_hidden=256, name="temporal_attention")(x)  # (B, 256)

    # 5) head de clasificare
    x = layers.Dense(head_hidden, activation=tf.nn.gelu, name="head_dense")(x)
    x = layers.Dropout(head_dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)

    return tf.keras.Model(inputs, outputs, name="video_vit_bilstm_attn")
