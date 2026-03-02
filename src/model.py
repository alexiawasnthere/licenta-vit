from dataclasses import dataclass
import tensorflow as tf
layers = tf.keras.layers


# config pentru vit (pe frame)
@dataclass(frozen=True)
class ViTConfig:
    img_size: int = 224
    patch_size: int = 32
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_dim: int = 256
    dropout: float = 0.1


class AddCLSToken(layers.Layer):
    # token cls trainable
    def __init__(self, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        # input_shape: (B, N, D)
        self.cls = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        b = tf.shape(x)[0]
        cls_tokens = tf.tile(self.cls, [b, 1, 1])
        return tf.concat([cls_tokens, x], axis=1)  # (B, N+1, D)


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
    # bloc standard transformer encoder
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
        # self-attention + residual
        y = self.norm1(x)
        y = self.attn(y, y, training=training)
        y = self.drop1(y, training=training)
        x = x + y

        # mlp + residual
        y = self.norm2(x)
        y = self.mlp1(y)
        y = self.drop2(y, training=training)
        y = self.mlp2(y)
        y = self.drop3(y, training=training)
        return x + y


def build_vit_encoder(cfg: ViTConfig) -> tf.keras.Model:
    # vit pentru un frame: (h,w,3) -> (d)
    if cfg.img_size % cfg.patch_size != 0:
        raise ValueError("img_size trebuie sa fie divizibil cu patch_size")

    num_patches = (cfg.img_size // cfg.patch_size) ** 2
    num_tokens = num_patches + 1  # + cls

    inputs = layers.Input(shape=(cfg.img_size, cfg.img_size, 3), name="frame")

    # patch embedding (conv stride=patch_size)
    x = layers.Conv2D(
        filters=cfg.embed_dim,
        kernel_size=cfg.patch_size,
        strides=cfg.patch_size,
        padding="valid",
        name="patch_embed"
    )(inputs)
    x = layers.Reshape((num_patches, cfg.embed_dim), name="flatten_patches")(x)

    # cls + positional
    x = AddCLSToken(cfg.embed_dim, name="add_cls")(x)
    x = AddPositionEmbedding(num_tokens, cfg.embed_dim, name="add_pos")(x)
    x = layers.Dropout(cfg.dropout)(x)

    # transformer blocks
    for i in range(cfg.depth):
        x = TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_dim, cfg.dropout, name=f"block_{i}")(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")(x)

    # folosim cls ca embedding final
    cls_out = layers.Lambda(lambda t: t[:, 0, :], name="cls_select")(x)
    return tf.keras.Model(inputs, cls_out, name="vit_encoder")


def build_video_vit_classifier(
    num_frames: int,
    img_size: int,
    num_classes: int,
    vit_cfg: ViTConfig | None = None,
    head_hidden: int = 256,
    head_dropout: float = 0.2,
) -> tf.keras.Model:
    # model video: vit pe fiecare frame + pooling temporal
    vit_cfg = vit_cfg or ViTConfig(img_size=img_size)
    vit_encoder = build_vit_encoder(vit_cfg)

    inputs = layers.Input(shape=(num_frames, img_size, img_size, 3), name="clip")

    # aplica vit pe frame-uri
    x = layers.TimeDistributed(vit_encoder, name="vit_per_frame")(inputs)  # (b, t, d)

    # pooling temporal (mean)
    x = layers.GlobalAveragePooling1D(name="temporal_mean")(x)  # (b, d)

    # head de clasificare
    x = layers.Dropout(head_dropout)(x)
    x = layers.Dense(head_hidden, activation=tf.nn.gelu)(x)
    x = layers.Dropout(head_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)

    return tf.keras.Model(inputs, outputs, name="video_vit_framewise")