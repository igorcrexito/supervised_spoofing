import tensorflow as tf

# DATA PARAMETERS
BATCH_SIZE = 4
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (4, 224, 224, 3)
NUM_CLASSES = 2

class VivitModel:
    def __init__(self, learning_rate: float, num_layers: int, num_heads: int, projection_dim: int, patch_size: (int, int, int),
                 layer_norm_eps: float, num_classes: int, input_shape: (int, int, int), summarize_model: bool):
        tubelet_embedder = TubeletEmbedding(embed_dim=projection_dim, patch_size=patch_size)
        positional_encoder = PositionalEncoder(embed_dim=projection_dim)

        self.summarize_model = summarize_model
        self.learning_rate = learning_rate
        self.model = self._create_model(transformer_layers=num_layers,
                                        tubelet_embedder=tubelet_embedder,
                                        positional_encoder=positional_encoder,
                                        num_heads=num_heads,
                                        embed_dim=projection_dim,
                                        layer_norm_eps=layer_norm_eps,
                                        input_shape=input_shape,
                                        num_classes=num_classes)



    def _create_model(self, tubelet_embedder, positional_encoder, input_shape,
            transformer_layers, num_heads, embed_dim, layer_norm_eps,num_classes,):

        inputs = tf.keras.layers.Input(shape=input_shape)
        patches = tubelet_embedder(inputs)
        encoded_patches = positional_encoder(patches)

        for _ in range(transformer_layers):
            # Layer normalization and MHSA
            x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
            )(x1, x1)

            # Skip connection
            x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

            # Layer Normalization and MLP
            x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                    tf.keras.layers.Dense(units=embed_dim, activation=tf.nn.gelu),
                ]
            )(x3)

            # Skip connection
            encoded_patches = tf.keras.layers.Add()([x3, x2])

        # Layer normalization and Global average pooling.
        representation = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
        representation = tf.keras.layers.GlobalAvgPool1D()(representation)

        # Classify outputs.
        outputs = tf.keras.layers.Dense(units=num_classes, activation="softmax")(representation)

        # Create the Keras model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        if self.summarize_model:
            model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            ],
        )

        return model


class TubeletEmbedding(tf.keras.layers.Layer):
        def __init__(self, embed_dim, patch_size, **kwargs):
            super().__init__(**kwargs)
            self.projection = tf.keras.layers.Conv3D(
                filters=embed_dim,
                kernel_size=patch_size,
                strides=patch_size,
                padding="VALID",
            )
            self.flatten = tf.keras.layers.Reshape(target_shape=(-1, embed_dim))

        def call(self, videos):
            projected_patches = self.projection(videos)
            flattened_patches = self.flatten(projected_patches)
            return flattened_patches

# Positional Encoder to add positional information to embeddings
class PositionalEncoder(tf.keras.layers.Layer):
        def __init__(self, embed_dim, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = embed_dim

        def build(self, input_shape):
            _, num_tokens, _ = input_shape
            self.position_embedding = tf.keras.layers.Embedding(
                input_dim=num_tokens, output_dim=self.embed_dim
            )
            self.positions = tf.range(start=0, limit=num_tokens, delta=1)

        def call(self, encoded_tokens):
            # Encode the positions and add it to the encoded tokens
            encoded_positions = self.position_embedding(self.positions)
            encoded_tokens = encoded_tokens + encoded_positions
            return encoded_tokens