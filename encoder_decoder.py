import functools
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class Decoder(tf.Module):
    """ Probabilistic decoder for `p(x_t | z_t)` """

    def __init__(self, base_depth=32, channels=3, scale=1.0, name=None):
        super(Decoder, self).__init__(name=name)
        self.scale = scale
        conv_transpose = functools.partial(
            tf.keras.layers.Conv2DTranspose, padding="SAME", activation=tf.nn.leaky_relu)
        self.conv_transpose1 = conv_transpose(8 * base_depth, 4, padding="VALID")
        self.conv_transpose2 = conv_transpose(4 * base_depth, 3, 2)
        self.conv_transpose3 = conv_transpose(2 * base_depth, 3, 2)
        self.conv_transpose4 = conv_transpose(base_depth, 3, 2)
        self.conv_transpose5 = conv_transpose(channels, 5, 2)

    def __call__(self, *inputs):
        if len(inputs) > 1:
            latent = tf.concat(inputs, axis=-1)
        else:
            latent, = inputs
        # (sample, N, T, latent)
        collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
        out = tf.reshape(latent, collapsed_shape)
        out = self.conv_transpose1(out)
        out = self.conv_transpose2(out)
        out = self.conv_transpose3(out)
        out = self.conv_transpose4(out)
        out = self.conv_transpose5(out)  # (sample*N*T, h, w, c)

        expanded_shape = tf.concat(
            [tf.shape(latent)[:-1], tf.shape(out)[1:]], axis=0)
        out = tf.reshape(out, expanded_shape)  # (sample, N, T, h, w, c)
        return tfd.Independent(
            distribution=tfd.Normal(loc=out, scale=self.scale),
            reinterpreted_batch_ndims=3)  # wrap (h, w, c)


class Encoder(tf.Module):
    """ Feature extractor """

    def __init__(self, base_depth=32, feature_size=32 * 8, name=None):
        super(Encoder, self).__init__(name=name)
        self.feature_size = feature_size
        conv = functools.partial(
            tf.keras.layers.Conv2D, padding="SAME", activation=tf.nn.leaky_relu)
        self.conv1 = conv(base_depth, 5, 2)
        self.conv2 = conv(2 * base_depth, 3, 2)
        self.conv3 = conv(4 * base_depth, 3, 2)
        self.conv4 = conv(8 * base_depth, 3, 2)
        self.conv5 = conv(8 * base_depth, 4, padding="VALID")

    def __call__(self, image):
        image_shape = tf.shape(image)[-3:]
        collapsed_shape = tf.concat(([-1], image_shape), axis=0)
        out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        expanded_shape = tf.concat((tf.shape(image)[:-3], [self.feature_size]), axis=0)
        return tf.reshape(out, expanded_shape)  # (sample, N, T, feature)


def _test_encoder():
    batch_size = 32
    horizon = 8
    image_shape = (64, 64, 3)

    encoder = Encoder()
    dummy_image = np.random.random(size=(batch_size, horizon,) + image_shape).astype(np.float32)
    out = encoder(image=dummy_image)
    print(out.shape)


def _test_decoder():
    batch_size = 32
    horizon = 8
    belief_size = 200
    state_size = 30

    decoder = Decoder()
    beliefs = tf.random.normal(shape=((batch_size - 1) * horizon, belief_size), dtype=tf.float32)
    posterior_states = tf.random.normal(shape=((batch_size - 1) * horizon, state_size), dtype=tf.float32)
    out = decoder(beliefs, posterior_states)
    print(out.sample().shape)


if __name__ == '__main__':
    import numpy as np
    from eager_setup import eager_setup

    eager_setup()

    print("=== Test Encoder ===")
    _test_encoder()

    print("=== Test Decoder ===")
    _test_decoder()
