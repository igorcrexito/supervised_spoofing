from tensorflow.keras.layers import Layer
import tensorflow as tf

class CustomBroadcastLayer(Layer):
    def __init__(self, cls_token, number_of_frames, num_patches, hidden_dim, **kwargs):  # Accept extra arguments
        super(CustomBroadcastLayer, self).__init__(**kwargs)  # Pass extra arguments to the parent class
        self.cls_token = tf.Variable(cls_token, trainable=True)  # Store as a TensorFlow Variable
        self.number_of_frames = number_of_frames
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim

    def call(self, x):
        batch_size = tf.shape(x)[0]
        return tf.broadcast_to(self.cls_token, [batch_size, self.number_of_frames, 1, self.hidden_dim])

    def get_config(self):
        config = super(CustomBroadcastLayer, self).get_config()
        config.update({
            'cls_token': self.cls_token.numpy().tolist(),  # Ensure compatibility
            'number_of_frames': self.number_of_frames,
            'num_patches': self.num_patches,
            'hidden_dim': self.hidden_dim
        })
        return config
