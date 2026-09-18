"""
Model architectures for ECG Arrhythmia Classification.

Four models for ablation study:
  1. BaselineCNN — pure convolutional model (time-domain only)
  2. CNNTransformer — CNN + Transformer encoder (time-domain only)
  3. FourierHybrid — dual-branch (time + frequency) CNN-Transformer
  4. FourierViTHybrid — dual-branch (time + frequency) with genuine ViT-style
     backbone: learnable [CLS] token, patch-style positional embeddings,
     and classification from the [CLS] token output

Architecture design principles:
  - Residual CNN blocks with BatchNormalization and GELU activation
    for stable gradient flow and modern activation characteristics
  - 1D Transformer / ViT Encoder with learnable positional embeddings
    and pre-norm architecture for stable training
  - FourierViTHybrid uses a genuine ViT-style design: CNN feature maps
    serve as "patches" (analogous to ViT's linear patch projections),
    a learnable [CLS] token is prepended, and classification uses the
    [CLS] output — making it a legitimate hybrid CNN-ViT
  - Feature fusion via concatenation + projection for time/frequency branches
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


# ============================================================
# Building blocks
# ============================================================

class ResidualConv1DBlock(layers.Layer):
    """
    Residual Conv1D block: Conv → BN → GELU → Conv → BN → Add → GELU

    Skip connection uses 1x1 conv when input/output dims differ.
    """

    def __init__(self, filters, kernel_size=5, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = layers.Conv1D(filters, kernel_size, padding="same",
                                   kernel_regularizer=regularizers.l2(1e-4))
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters, kernel_size, padding="same",
                                   kernel_regularizer=regularizers.l2(1e-4))
        self.bn2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout_rate)
        self.skip_conv = None  # built dynamically
        self.skip_bn = None

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.skip_conv = layers.Conv1D(self.filters, 1, padding="same")
            self.skip_bn = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.keras.activations.gelu(x)
        x = self.dropout(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # Skip connection
        skip = inputs
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)
            skip = self.skip_bn(skip, training=training)

        x = x + skip
        x = tf.keras.activations.gelu(x)
        return x


class PositionalEncoding(layers.Layer):
    """Learnable 1D positional encoding."""

    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.max_len, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_embed[:, :seq_len, :]


class TransformerEncoderBlock(layers.Layer):
    """
    Pre-norm Transformer Encoder block.

    Pre-norm (LN before attention/FFN) is more stable for training
    than post-norm, especially without extensive warmup schedules.
    """

    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(
            key_dim=d_model // num_heads,
            num_heads=num_heads,
            dropout=dropout_rate,
        )
        self.dropout1 = layers.Dropout(dropout_rate)

        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu",
                         kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model,
                         kernel_regularizer=regularizers.l2(1e-4)),
        ])
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        # Pre-norm MHA
        x = self.ln1(inputs)
        attn_output = self.mha(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = inputs + attn_output

        # Pre-norm FFN
        y = self.ln2(x)
        ffn_output = self.ffn(y, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return x + ffn_output


class AttentionPooling(layers.Layer):
    """
    Attention-weighted pooling: learns which time steps are most important.

    More expressive than GlobalAveragePooling for sequential data.
    """

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.Dense(1)
        self.d_model = d_model

    def call(self, inputs):
        # inputs: (batch, seq_len, d_model)
        weights = self.attention(inputs)  # (batch, seq_len, 1)
        weights = tf.nn.softmax(weights, axis=1)
        weighted = inputs * weights
        return tf.reduce_sum(weighted, axis=1)  # (batch, d_model)


class CLSToken(layers.Layer):
    """
    Learnable [CLS] token — the hallmark of ViT-style architectures.

    Prepends a single learnable token to the sequence. After passing
    through the Transformer encoder, the [CLS] token's output is used
    for classification (instead of pooling over the full sequence).

    This is what distinguishes a genuine ViT from a plain Transformer
    applied to features: the model learns a dedicated summary vector
    via attention over all sequence positions.
    """

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcast = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_broadcast, inputs], axis=1)


# ============================================================
# CNN Feature Extractor (shared by all models)
# ============================================================

def build_cnn_branch(input_shape, filters_list, kernel_size=5,
                     dropout_rate=0.2, name_prefix="cnn"):
    """
    Build a residual CNN feature extractor.

    Architecture:
      ResidualConv1D(32) → MaxPool → ResidualConv1D(64) → MaxPool → ResidualConv1D(128)
    """
    inp = layers.Input(shape=input_shape, name=f"{name_prefix}_input")
    x = inp

    for i, filters in enumerate(filters_list):
        x = ResidualConv1DBlock(
            filters, kernel_size, dropout_rate,
            name=f"{name_prefix}_res_block_{i}"
        )(x)
        if i < len(filters_list) - 1:
            x = layers.MaxPooling1D(pool_size=2, name=f"{name_prefix}_pool_{i}")(x)

    return inp, x


# ============================================================
# Model 1: Baseline CNN
# ============================================================

def build_baseline_cnn(signal_length=186, num_classes=5,
                       filters_list=None, dropout_rate=0.3):
    """
    Pure CNN baseline — no Transformer, no FFT.

    Used as the control in the ablation study.
    """
    if filters_list is None:
        filters_list = [32, 64, 128]

    inp, x = build_cnn_branch(
        (signal_length, 1), filters_list, dropout_rate=dropout_rate,
        name_prefix="time"
    )

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="gelu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="gelu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate)(x)

    output = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inp, outputs=output, name="BaselineCNN")
    return model


# ============================================================
# Model 2: CNN + Transformer (time-domain only)
# ============================================================

def build_cnn_transformer(signal_length=186, num_classes=5,
                          filters_list=None, num_heads=4,
                          ff_dim=128, num_transformer_layers=2,
                          dropout_rate=0.3):
    """
    CNN + Transformer Encoder — time-domain only.

    CNN extracts local morphological features, Transformer captures
    global temporal dependencies across the beat.
    """
    if filters_list is None:
        filters_list = [32, 64, 128]

    inp, x = build_cnn_branch(
        (signal_length, 1), filters_list, dropout_rate=dropout_rate,
        name_prefix="time"
    )

    d_model = filters_list[-1]

    # Positional encoding
    x = PositionalEncoding(max_len=signal_length, d_model=d_model)(x)

    # Transformer encoder stack
    for i in range(num_transformer_layers):
        x = TransformerEncoderBlock(
            d_model=d_model, num_heads=num_heads,
            ff_dim=ff_dim, dropout_rate=dropout_rate,
            name=f"transformer_{i}"
        )(x)

    # Final layer norm
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Attention pooling
    x = AttentionPooling(d_model)(x)

    x = layers.Dense(128, activation="gelu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate)(x)

    output = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inp, outputs=output, name="CNN_Transformer")
    return model


# ============================================================
# Model 3: Fourier Hybrid — Time + Frequency CNN-Transformer
# ============================================================

def build_fourier_hybrid(signal_length=186, fft_length=94,
                         num_classes=5, filters_list=None,
                         num_heads=4, ff_dim=128,
                         num_transformer_layers=2, dropout_rate=0.3):
    """
    Dual-branch hybrid architecture:

      Branch A (Time-domain):
        ECG signal → Residual CNN → local morphology features

      Branch B (Frequency-domain):
        FFT magnitude → Residual CNN → spectral features

      Fusion:
        Concatenate → Dense projection → Positional encoding
        → Transformer Encoder stack → Attention pooling → Classifier

    This is the proposed model. The dual-branch design allows the
    Transformer to attend over both temporal morphology and spectral
    content simultaneously, providing richer feature representations.
    """
    if filters_list is None:
        filters_list = [32, 64, 128]

    # ---- Time-domain branch ----
    time_inp, time_feat = build_cnn_branch(
        (signal_length, 1), filters_list, dropout_rate=dropout_rate,
        name_prefix="time"
    )

    # ---- Frequency-domain branch ----
    fft_inp, fft_feat = build_cnn_branch(
        (fft_length, 1), filters_list, dropout_rate=dropout_rate,
        name_prefix="freq"
    )

    # ---- Feature Fusion ----
    # Project both branches to same dimension, then concatenate along sequence
    d_model = filters_list[-1]

    time_proj = layers.Dense(d_model, name="time_projection")(time_feat)
    freq_proj = layers.Dense(d_model, name="freq_projection")(fft_feat)

    # Concatenate along the sequence dimension
    fused = layers.Concatenate(axis=1, name="fusion")([time_proj, freq_proj])

    # Positional encoding over the fused sequence
    fused = PositionalEncoding(
        max_len=signal_length + fft_length, d_model=d_model
    )(fused)

    # ---- Transformer Encoder ----
    x = fused
    for i in range(num_transformer_layers):
        x = TransformerEncoderBlock(
            d_model=d_model, num_heads=num_heads,
            ff_dim=ff_dim, dropout_rate=dropout_rate,
            name=f"transformer_{i}"
        )(x)

    # Final layer norm
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Attention pooling
    x = AttentionPooling(d_model)(x)

    # ---- Classification head ----
    x = layers.Dense(128, activation="gelu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate)(x)

    output = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=[time_inp, fft_inp], outputs=output,
                  name="Fourier_Hybrid_CNN_Transformer")
    return model


# ============================================================
# Model 4: Fourier ViT Hybrid — Genuine ViT-style backbone
# ============================================================

def build_fourier_vit_hybrid(signal_length=186, fft_length=94,
                              num_classes=5, filters_list=None,
                              num_heads=4, ff_dim=128,
                              num_transformer_layers=2, dropout_rate=0.3):
    """
    Dual-branch hybrid with a genuine ViT-style backbone.

    This model adapts the Vision Transformer (ViT) architecture to 1D
    ECG signals. The key ViT elements that distinguish this from a
    plain Transformer:

      1. **Patch-like representation**: CNN feature maps serve as the
         "patches" (analogous to ViT's linear projection of image patches)
      2. **[CLS] token**: A learnable class token is prepended to the
         sequence — the model learns to aggregate information into it
      3. **Classification from [CLS]**: The final classification uses
         ONLY the [CLS] token output, not pooling over the sequence

    Architecture:
      Branch A (Time): ECG → Residual CNN → feature "patches"
      Branch B (Freq): FFT → Residual CNN → feature "patches"
      Fusion: Concat → Dense → [CLS] + Positional Encoding
              → ViT Encoder × N → CLS output → Classifier
    """
    if filters_list is None:
        filters_list = [32, 64, 128]

    # ---- Time-domain branch ----
    time_inp, time_feat = build_cnn_branch(
        (signal_length, 1), filters_list, dropout_rate=dropout_rate,
        name_prefix="time"
    )

    # ---- Frequency-domain branch ----
    fft_inp, fft_feat = build_cnn_branch(
        (fft_length, 1), filters_list, dropout_rate=dropout_rate,
        name_prefix="freq"
    )

    # ---- Feature Fusion ----
    d_model = filters_list[-1]

    time_proj = layers.Dense(d_model, name="time_projection")(time_feat)
    freq_proj = layers.Dense(d_model, name="freq_projection")(fft_feat)

    # Concatenate along the sequence dimension (patches from both branches)
    fused = layers.Concatenate(axis=1, name="fusion")([time_proj, freq_proj])

    # ---- ViT-specific: Prepend [CLS] token ----
    fused = CLSToken(d_model, name="cls_token")(fused)

    # Positional encoding over [CLS] + fused sequence
    max_seq_len = signal_length + fft_length + 1  # +1 for [CLS]
    fused = PositionalEncoding(
        max_len=max_seq_len, d_model=d_model
    )(fused)

    # ---- ViT Encoder (same Transformer blocks, ViT-style usage) ----
    x = fused
    for i in range(num_transformer_layers):
        x = TransformerEncoderBlock(
            d_model=d_model, num_heads=num_heads,
            ff_dim=ff_dim, dropout_rate=dropout_rate,
            name=f"vit_encoder_{i}"
        )(x)

    # Final layer norm
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # ---- ViT-specific: Classify from [CLS] token only ----
    # Extract the [CLS] token (position 0)
    cls_output = x[:, 0, :]  # (batch, d_model)

    # ---- Classification head ----
    cls_output = layers.Dense(128, activation="gelu",
                              kernel_regularizer=regularizers.l2(1e-4))(cls_output)
    cls_output = layers.Dropout(dropout_rate)(cls_output)
    cls_output = layers.Dense(64, activation="gelu",
                              kernel_regularizer=regularizers.l2(1e-4))(cls_output)
    cls_output = layers.Dropout(dropout_rate)(cls_output)

    output = layers.Dense(num_classes, activation="softmax", name="output")(cls_output)

    model = Model(inputs=[time_inp, fft_inp], outputs=output,
                  name="Fourier_ViT_Hybrid")
    return model


# ============================================================
# Model builder utility
# ============================================================

def build_model(model_type: str, signal_length: int = 186,
                fft_length: int = 94, num_classes: int = 5, **kwargs):
    """
    Build a model by name.

    Args:
        model_type: one of "baseline_cnn", "cnn_transformer",
                    "fourier_hybrid", "fourier_vit_hybrid"
        signal_length: number of time-domain samples
        fft_length: number of FFT bins (signal_length // 2 + 1)
        num_classes: number of output classes

    Returns:
        Compiled Keras Model
    """
    builders = {
        "baseline_cnn": lambda: build_baseline_cnn(
            signal_length=signal_length, num_classes=num_classes, **kwargs
        ),
        "cnn_transformer": lambda: build_cnn_transformer(
            signal_length=signal_length, num_classes=num_classes, **kwargs
        ),
        "fourier_hybrid": lambda: build_fourier_hybrid(
            signal_length=signal_length, fft_length=fft_length,
            num_classes=num_classes, **kwargs
        ),
        "fourier_vit_hybrid": lambda: build_fourier_vit_hybrid(
            signal_length=signal_length, fft_length=fft_length,
            num_classes=num_classes, **kwargs
        ),
    }

    if model_type not in builders:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Choose from {list(builders.keys())}")

    return builders[model_type]()
