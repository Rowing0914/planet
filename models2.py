from typing import Optional, List
import tensorflow as tf
from eager_setup import eager_setup

eager_setup()


class TransitionModel(tf.Module):
    def __init__(self,
                 belief_size,
                 state_size,
                 action_size,
                 hidden_size,
                 embedding_size,
                 activation_function="relu",
                 min_std_dev=0.1):
        super(TransitionModel, self).__init__()
        self.act_fn = None
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = tf.keras.layers.Dense(belief_size)
        self.rnn = tf.keras.layers.GRUCell(belief_size, belief_size)


class VisualObservationModel(tf.Module):
    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.embedding_size = embedding_size
        self.fc1 = tf.keras.layers.Dense(embedding_size, activation=activation_function)
        self.conv1 = tf.keras.layers.Conv2DTranspose(128, 5, 2, activation=activation_function)
        self.conv2 = tf.keras.layers.Conv2DTranspose(64, 5, 2, activation=activation_function)
        self.conv3 = tf.keras.layers.Conv2DTranspose(32, 6, 2, activation=activation_function)
        self.conv4 = tf.keras.layers.Conv2DTranspose(3, 6, 2, activation=activation_function)

    def __call__(self, belief, state):
        inputs = tf.concat([belief, state], axis=1)  # No nonlinearity here
        hidden = self.fc1(inputs)
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.conv1(hidden)
        hidden = self.conv2(hidden)
        hidden = self.conv3(hidden)
        observation = self.conv4(hidden)
        return observation


class RewardModel(tf.Module):
    def __init__(self, hidden_size, activation_function='relu'):
        super(RewardModel, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation=activation_function)
        self.fc2 = tf.keras.layers.Dense(hidden_size, activation=activation_function)
        self.fc3 = tf.keras.layers.Dense(1, activation=activation_function)

    def __call__(self, belief, state):
        inputs = tf.concat([belief, state], axis=1)  # No nonlinearity here
        hidden = self.fc2(inputs)
        reward = self.fc3(hidden)
        return reward


class VisualEncoder(tf.Module):
    def __init__(self, embedding_size, activation_function='relu'):
        super(VisualEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.conv1 = tf.keras.layers.Conv2D(32, 4, 2, activation=activation_function)
        self.conv2 = tf.keras.layers.Conv2D(64, 4, 2, activation=activation_function)
        self.conv3 = tf.keras.layers.Conv2D(128, 4, 2, activation=activation_function)
        self.conv4 = tf.keras.layers.Conv2D(256, 4, 2, activation=activation_function)
        self.flatten = tf.keras.layers.Flatten()

        if embedding_size == 1024:
            self.fc = lambda x: tf.identity(x)
        else:
            self.fc = tf.keras.layers.Dense(1024, embedding_size, activation=activation_function)

    def __call__(self, observation):
        hidden = self.conv1(observation)
        hidden = self.conv2(hidden)
        hidden = self.conv3(hidden)
        hidden = self.conv4(hidden)
        hidden = self.flatten(hidden)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


def roll_out(env, buffer):
    total_reward = 0
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward

        if done: break

    env.close()
    print("Total Reward: {}".format(total_reward))
    return buffer


def _test_encoder():
    """ test function for the encoder """
    import numpy as np
    from dm_control2gym.util import make_dm2gym_env_obs
    from memory2 import ReplayBuffer

    model = VisualEncoder(embedding_size=1024)
    buffer = ReplayBuffer(size=1000, n_step=50)
    env = make_dm2gym_env_obs(env_name="cartpole_balance", num_repeat_action=4)
    buffer = roll_out(env=env, buffer=buffer)
    obses_t, actions, rewards, obses_tp1, dones = buffer.sample(batch_size=50)
    obses_t = obses_t / 255.0  # normalise the images
    newshape = (obses_t.shape[0] * obses_t.shape[1],) + obses_t.shape[2:]  # No Latent Overshooting!!
    obses_t = np.reshape(obses_t, newshape=newshape)  # reshape the images
    output = model(obses_t.astype(np.float32))  # get the encoded latent states
    print(output)


def _test_observation_model():
    import numpy as np
    from dm_control2gym.util import make_dm2gym_env_obs
    from memory2 import ReplayBuffer

    batch_size = 50
    belief_size = 200
    state_size = 30
    embedding_size = 1024

    buffer = ReplayBuffer(size=1000, n_step=50)
    env = make_dm2gym_env_obs(env_name="cartpole_balance", num_repeat_action=4)
    model = VisualObservationModel(embedding_size=embedding_size)

    beliefs = tf.random.normal(shape=(49, batch_size, belief_size))
    posterior_states = tf.random.normal(shape=(49, batch_size, state_size))

    buffer = roll_out(env=env, buffer=buffer)
    obses_t, actions, rewards, obses_tp1, dones = buffer.sample(batch_size=50)
    obses_t = obses_t / 255.0  # normalise the images
    newshape = (obses_t.shape[0] * obses_t.shape[1],) + obses_t.shape[2:]  # No Latent Overshooting!!
    obses_t = np.reshape(obses_t, newshape=newshape)  # reshape the images
    output = model(obses_t.astype(np.float32))  # get the encoded latent states
    print(output)


if __name__ == '__main__':
    _test_encoder()
