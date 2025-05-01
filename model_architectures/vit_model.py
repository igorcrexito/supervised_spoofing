from keras.layers import Input, Conv2D, Conv3D, MaxPooling3D, UpSampling2D, Concatenate, LayerNormalization, ZeroPadding2D, \
    BatchNormalization, Flatten, Dense, Reshape, DepthwiseConv2D, Add, Dropout, MultiHeadAttention, Rescaling, Activation, \
LayerNormalization, MultiHeadAttention, Lambda
from layers.custom_layers import CustomBroadcastLayer
from keras.models import Model
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam
import sys
sys.path.insert(0, '..')

class VitModel:
    def __init__(self, number_of_frames: int, patch_size: int, image_size: int, num_heads: int, mlp_dim: int, hidden_dim: int,
                 summarize_model: bool, pre_trained_path: str = None):
        self.summarize_model = summarize_model
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.number_of_frames = number_of_frames

        if pre_trained_path is not None:
            self.model = keras.models.load_model(pre_trained_path, custom_objects={"CustomBroadcastLayer": CustomBroadcastLayer})
        else:
            self.model = self._create_model()

    def _create_model(self):
        num_patches = (self.image_size // self.patch_size) ** 2
        input_layer = Input(shape=(self.number_of_frames, self.image_size, self.image_size, 3))

        # Patch embedding
        x = Conv3D(self.hidden_dim, kernel_size=(1, self.patch_size, self.patch_size),
                   strides=(1, self.patch_size, self.patch_size), padding='valid')(input_layer)
        x = Reshape(target_shape=(self.number_of_frames, num_patches, self.hidden_dim))(x)

        # Positional embedding (Fixed shape to match input tensor)
        pos_embedding = tf.Variable(tf.random.normal([1, self.number_of_frames, num_patches + 1, self.hidden_dim]),
                                    trainable=True,
                                    name="pos_embedding")

        cls_token = tf.Variable(tf.random.normal([1, 1, 1, self.hidden_dim]), trainable=True, name="cls_token")

        # Use Custom Layer for Broadcasting
        cls_tokens = CustomBroadcastLayer(cls_token, self.number_of_frames, num_patches, self.hidden_dim)(x)

        # Concatenate cls_tokens and x along the num_patches axis (axis=2)
        x = Concatenate(axis=2)([cls_tokens, x])  # Corrected axis

        # Add positional embeddings (ensuring shape matches)
        x = x + pos_embedding

        # Transformer Encoder
        for _ in range(2):
            skip = x
            x = LayerNormalization()(x)
            x = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.hidden_dim // self.num_heads)(x, x)  # Fixed key_dim
            x = Dropout(0.1)(x)
            x = skip + x

            skip = x
            x = LayerNormalization()(x)
            x = Dense(self.mlp_dim, activation="gelu")(x)
            x = Dense(self.hidden_dim)(x)
            x = Dropout(0.1)(x)
            x = skip + x

        # Classification Head - Extracting CLS token correctly
        x = LayerNormalization()(x[:, :, 0, :])  # Fixed indexing for CLS token
        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)
        output = Dense(2, activation="softmax")(x)

        model = Model(input_layer, output, name="VisionTransformer")

        if self.summarize_model:
            model.summary()

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def get_config(self):
        """
        Returns the configuration of the model as a dictionary.
        This is required for serialization and saving.
        """
        config = {
            "number_of_frames": self.number_of_frames,
            "patch_size": self.patch_size,
            "image_size": self.image_size,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "hidden_dim": self.hidden_dim,
            "summarize_model": self.summarize_model
        }
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates an instance of the model from a configuration dictionary.
        This is required for deserialization.
        """
        return cls(**config)