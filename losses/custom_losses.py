from keras import backend as K
import tensorflow as tf

def root_mean_squared_error(y_true, y_pred):
    y_true = float(y_true)
    y_pred = float(y_pred)
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def focal_loss_for_regression(gamma=2.0, alpha=0.25):
    gamma = tf.constant(gamma, dtype=tf.float32)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def focal_loss(y_true, y_pred):
        # Cast y_true and y_pred to tf.float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate the absolute error
        abs_error = tf.abs(y_true - y_pred)

        # Calculate the focal loss
        loss = tf.pow(abs_error, gamma) * tf.math.log1p(abs_error)

        # Apply the alpha scaling factor
        loss = alpha * loss

        return tf.reduce_mean(loss, axis=-1)

    return focal_loss