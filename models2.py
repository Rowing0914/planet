import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def reshape(x, tf_flg=True):
    """ Reshape an input to remove the latent overshooting

    :param x: batch-size x time-horizon x other shape(e.g., image)
    :param tf_flg: data type of x
    """
    if tf_flg:
        print((int(x.shape[0] * x.shape[1]),), x.shape[2:])
        newshape = (int(x.shape[0] * x.shape[1]),) + x.shape[2:]  # integrate the time-horizon into the batch-size
        x = tf.reshape(x, shape=newshape)
        return x
    else:
        newshape = (int(x.shape[0] * x.shape[1]),) + x.shape[2:]  # integrate the time-horizon into the batch-size
        x = np.reshape(x, newshape=newshape)
        return x


class TransitionModel(tf.Module):
    def __init__(self,
                 belief_size=200,
                 state_size=30,
                 hidden_size=200,
                 min_std_dev=0.1):
        super(TransitionModel, self).__init__()
        self.act_fn = None
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = tf.keras.layers.Dense(belief_size, activation='relu')
        # self.rnn = tf.keras.layers.GRUCell(belief_size, activation='tanh')
        self.rnn = tf.compat.v1.nn.rnn_cell.GRUCell(belief_size, activation='tanh')
        self.fc_embed_belief_prior = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc_state_prior = tf.keras.layers.Dense(2 * state_size, activation='relu')
        self.fc_embed_belief_posterior = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc_state_posterior = tf.keras.layers.Dense(2 * state_size, activation='relu')

    def __call__(self, prev_state, actions, prev_belief, obs=None, dones=None):
        """ Main cycle of Transition Model using RNN

        :param prev_state: horizon x batch x state_size
        :param actions: horizon x batch x action_size
        :param prev_belief: horizon x batch x belief_size
        :param obs: batch x feat_size(after encoded by encoder)
        :param dones: horizon x batch x 1(or None)
        """
        # instantiate the TensorArray to go through the horizon
        T = actions.shape[0] + 1
        # beliefs = tf.TensorArray(dtype=tf.float32, size=T)
        # prior_states = tf.TensorArray(dtype=tf.float32, size=T)
        # prior_means = tf.TensorArray(dtype=tf.float32, size=T)
        # prior_std_devs = tf.TensorArray(dtype=tf.float32, size=T)
        # posterior_states = tf.TensorArray(dtype=tf.float32, size=T)
        # posterior_means = tf.TensorArray(dtype=tf.float32, size=T)
        # posterior_std_devs = tf.TensorArray(dtype=tf.float32, size=T)

        # set the previous beliefs/states
        # beliefs = beliefs.write(index=0, value=prev_belief)
        # prior_states = prior_states.write(index=0, value=prev_state)
        # posterior_states = posterior_states.write(index=0, value=prev_state)

        beliefs = [tf.zeros_like(prev_belief, dtype=tf.float32)] * T
        prior_states = [tf.zeros_like(prev_state, dtype=tf.float32)] * T
        prior_means = [tf.zeros(shape=())] * T
        prior_std_devs = [tf.zeros(shape=())] * T
        posterior_states = [tf.zeros_like(prev_state, dtype=tf.float32)] * T
        posterior_means = [tf.zeros(shape=())] * T
        posterior_std_devs = [tf.zeros(shape=())] * T

        # set the previous beliefs/states
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state

        for t in range(T - 1):
            # Select appropriate previous state
            _state = prior_states[t] if obs is None else posterior_states[t]  # batch x state_size

            # Mask if previous transition was terminal
            _state = _state if dones is None else _state * dones[t]  # batch x state_size

            # Compute belief (deterministic hidden state)
            hidden = self.fc_embed_state_action(tf.concat([_state, actions[t]], axis=-1))  # batch x belief
            beliefs[t + 1], _ = self.rnn(hidden, beliefs[t])

            # Compute state prior by applying transition dynamics
            hidden = self.fc_embed_belief_prior(beliefs[t + 1])
            prior_means[t + 1], _prior_std_dev = tf.split(self.fc_state_prior(hidden), num_or_size_splits=2, axis=1)
            prior_std_devs[t + 1] = tf.math.softplus(_prior_std_dev) + self.min_std_dev
            noise = tf.random.normal(shape=prior_means[t + 1].shape)
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * noise

            if obs is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.fc_embed_belief_posterior(tf.concat([beliefs[t + 1], obs[t_ + 1]], axis=1))
                posterior_means[t + 1], _posterior_std_dev = tf.split(self.fc_state_posterior(hidden), 2, axis=1)
                posterior_std_devs[t + 1] = tf.math.softplus(_posterior_std_dev) + self.min_std_dev
                noise = tf.random.normal(shape=posterior_means[t + 1].shape)
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * noise

        # Return new hidden states
        hidden = [tf.stack(beliefs[1:], axis=0),
                  tf.stack(prior_states[1:], axis=0),
                  tf.stack(prior_means[1:], axis=0),
                  tf.stack(prior_std_devs[1:], axis=0)]
        if obs is not None:
            hidden += [tf.stack(posterior_states[1:], axis=0),
                       tf.stack(posterior_means[1:], axis=0),
                       tf.stack(posterior_std_devs[1:], axis=0)]
        return hidden


class RewardModel(tf.Module):
    def __init__(self, hidden_size=200, activation_function='relu'):
        super(RewardModel, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation=activation_function)
        self.fc2 = tf.keras.layers.Dense(hidden_size, activation=activation_function)
        self.fc3 = tf.keras.layers.Dense(1, activation=activation_function)

    def __call__(self, belief, state):
        inputs = tf.concat([belief, state], axis=1)  # No nonlinearity here
        hidden = self.fc2(inputs)
        reward = self.fc3(hidden)
        return reward


class Decoder(tf.Module):
    """ Probabilistic decoder for `p(x_t | z_t)` """

    def __init__(self, embedding_size=1024, scale=1.0, name=None):
        super(Decoder, self).__init__(name=name)
        self.scale = scale
        self.fc1 = tf.keras.layers.Dense(embedding_size, activation=tf.nn.relu)
        self.deconv1 = tf.keras.layers.Conv2DTranspose(128, 5, strides=2, activation=tf.nn.relu)
        self.deconv2 = tf.keras.layers.Conv2DTranspose(64, 5, strides=2, activation=tf.nn.relu)
        self.deconv3 = tf.keras.layers.Conv2DTranspose(32, 6, strides=2, activation=tf.nn.relu)
        self.deconv4 = tf.keras.layers.Conv2DTranspose(3, 6, strides=2)

    def __call__(self, *inputs):
        if len(inputs) > 1:
            latent = tf.concat(inputs, axis=-1)
        else:
            latent, = inputs
        # (sample, N, T, latent)
        collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
        out = tf.reshape(latent, collapsed_shape)
        out = self.fc1(out)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)

        expanded_shape = tf.concat(
            [tf.shape(latent)[:-1], tf.shape(out)[1:]], axis=0)
        out = tf.reshape(out, expanded_shape)  # (sample, N, T, h, w, c)
        return tfd.Independent(
            distribution=tfd.Normal(loc=out, scale=self.scale),
            reinterpreted_batch_ndims=3)  # wrap (h, w, c)


class Encoder(tf.Module):
    """ Feature extractor """

    def __init__(self, embedding_size=1024, name=None):
        super(Encoder, self).__init__(name=name)
        self._embedding_size = embedding_size
        self.conv1 = tf.keras.layers.Conv2D(32, 4, 2, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, 4, 2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(128, 4, 2, activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(256, 4, 2, activation='relu')
        self.flatten = tf.keras.layers.Flatten()

        if embedding_size == 1024:
            self.fc = lambda x: tf.identity(x)
        else:
            self.fc = tf.keras.layers.Dense(1024, embedding_size, activation="relu")

    def __call__(self, image):
        image_shape = tf.shape(image)[-3:]
        collapsed_shape = tf.concat(([-1], image_shape), axis=0)
        out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.flatten(out)
        return self.fc(out)
