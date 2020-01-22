import tensorflow as tf
from eager_setup import eager_setup
from models2 import Encoder, Decoder, TransitionModel, RewardModel, reshape

eager_setup()


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


def _test_encoder(batch_size=32, horizon=8):
    from dm_control2gym.util import make_dm2gym_env_obs
    from memory2 import ReplayBuffer
    from get_gpu_info import print_gpu_info

    encoder = Encoder()
    buffer = ReplayBuffer(size=1000, n_step=horizon)
    env = make_dm2gym_env_obs(env_name="cartpole_balance", num_repeat_action=4)
    buffer = roll_out(env=env, buffer=buffer)
    obses_t, actions, rewards, obses_tp1, dones = buffer.sample(batch_size=batch_size)
    obses_t = obses_t / 255.0  # normalise the images
    obses_t = reshape(x=obses_t, tf_flg=False)
    out = encoder(image=obses_t.astype(np.float32))
    print_gpu_info(gpu_id=0)


def _test_decoder(batch_size=32, horizon=8, belief_size=200, state_size=30):
    from get_gpu_info import print_gpu_info

    decoder = Decoder()
    beliefs = tf.random.normal(shape=((batch_size - 1) * horizon, belief_size), dtype=tf.float32)
    posterior_states = tf.random.normal(shape=((batch_size - 1) * horizon, state_size), dtype=tf.float32)
    out = decoder(beliefs, posterior_states)
    print_gpu_info(gpu_id=0)
    print(out.sample().shape)


def _test_reward_model(batch_size=32, horizon=8, belief_size=200, state_size=30):
    from get_gpu_info import print_gpu_info

    model = RewardModel()
    beliefs = tf.random.normal(shape=((batch_size - 1) * horizon, belief_size), dtype=tf.float32)
    posterior_states = tf.random.normal(shape=((batch_size - 1) * horizon, state_size), dtype=tf.float32)
    out = model(beliefs, posterior_states)
    print_gpu_info(gpu_id=0)
    print(out.shape)


def _test_transition_model(batch_size=32, horizon=8, belief_size=200, state_size=30):
    from dm_control2gym.util import make_dm2gym_env_obs
    from get_gpu_info import print_gpu_info
    from memory2 import ReplayBuffer

    encoder = Encoder()
    buffer = ReplayBuffer(size=1000, n_step=horizon)
    env = make_dm2gym_env_obs(env_name="cartpole_balance", num_repeat_action=4)
    model = TransitionModel()

    # create dummy data
    init_belief = tf.random.normal(shape=(batch_size, belief_size), dtype=tf.float32)
    init_state = tf.random.normal(shape=(batch_size, state_size), dtype=tf.float32)

    # roll-out to collect samples
    buffer = roll_out(env=env, buffer=buffer)
    obses_t, actions, rewards, obses_tp1, dones = buffer.sample(batch_size=batch_size)
    obses_t = obses_t / 255.0  # normalise the images
    obses_t = reshape(x=obses_t, tf_flg=False)

    # encode the images into the latent space
    feat = encoder(image=obses_t.astype(np.float32))
    feat = tf.reshape(feat, shape=(horizon, batch_size, -1))

    # transpose the data into: horizon x batch x dim_data
    dones = np.transpose(dones, axes=(1, 0))
    actions = np.transpose(actions, axes=(1, 0, 2))

    dones = np.tile(dones[..., np.newaxis], reps=(1, 1, state_size))

    # apply the forward dynamics over the horizon at once
    out = model(prev_state=init_state,
                actions=actions,
                prev_belief=init_belief,
                obs=feat,
                dones=dones)
    belief, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = out
    print_gpu_info(gpu_id=0)
    print(belief.shape,
          prior_states.shape,
          prior_means.shape,
          prior_std_devs.shape,
          posterior_states.shape,
          posterior_means.shape,
          posterior_std_devs.shape)


def _test_train(batch_size=32, horizon=8, belief_size=200, state_size=30, free_nats=3.0):
    import tensorflow_probability as tfp
    from dm_control2gym.util import make_dm2gym_env_obs
    from get_gpu_info import print_gpu_info
    from memory2 import ReplayBuffer

    tfd = tfp.distributions

    encoder = Encoder()
    decoder = Decoder()
    reward_model = RewardModel()
    model = TransitionModel()
    optimiser = tf.compat.v1.train.AdamOptimizer()
    buffer = ReplayBuffer(size=1000, n_step=horizon)
    env = make_dm2gym_env_obs(env_name="cartpole_balance", num_repeat_action=4)

    # create dummy data
    init_belief = tf.random.normal(shape=(batch_size, belief_size), dtype=tf.float32)
    init_state = tf.random.normal(shape=(batch_size, state_size), dtype=tf.float32)

    # roll-out to collect samples
    buffer = roll_out(env=env, buffer=buffer)
    obses_t, actions, rewards, obses_tp1, dones = buffer.sample(batch_size=batch_size)
    obses_t = obses_t / 255.0  # normalise the images
    obses_t = reshape(x=obses_t, tf_flg=False)

    # transpose the data into: horizon x batch x dim_data
    dones = np.transpose(dones, axes=(1, 0))
    rewards = np.transpose(rewards, axes=(1, 0))
    rewards = reshape(x=rewards, tf_flg=False)[..., np.newaxis]
    actions = np.transpose(actions, axes=(1, 0, 2))

    dones = np.tile(dones[..., np.newaxis], reps=(1, 1, state_size))

    with tf.GradientTape() as tape:
        # encode the images into the latent space
        feat = encoder(image=obses_t.astype(np.float32))
        feat = tf.reshape(feat, shape=(horizon, batch_size, -1))

        # apply the forward dynamics over the horizon at once
        out = model(prev_state=init_state,
                    actions=actions,
                    prev_belief=init_belief,
                    obs=feat,
                    dones=dones)
        belief, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = out
        belief, posterior_states = reshape(belief), reshape(posterior_states)
        reconst_loss = tf.compat.v1.losses.mean_squared_error(decoder(belief, posterior_states), obses_t)
        reward_loss = tf.compat.v1.losses.mean_squared_error(reward_model(belief, posterior_states), rewards)
        posterior = tfd.Normal(loc=posterior_means, scale=posterior_std_devs)
        prior = tfd.Normal(loc=prior_means, scale=prior_std_devs)
        kl_div = tfd.kl_divergence(posterior, prior)
        kl_loss = tf.math.maximum(tf.math.reduce_sum(kl_div, axis=-1), free_nats)
        kl_loss = tf.math.reduce_mean(kl_loss)
        loss = reconst_loss + reward_loss + kl_loss

    grads = tape.gradient(loss, [encoder.trainable_variables, decoder.trainable_variables, model.trainable_variables])
    # TODO: applying gradients causes an error, need to fix this!!
    optimiser.apply_gradients(zip(grads, [encoder.trainable_variables, decoder.trainable_variables, model.trainable_variables]))

    print_gpu_info(gpu_id=0)
    print(loss)


if __name__ == '__main__':
    import numpy as np
    from eager_setup import eager_setup

    batch_size, horizon = 32, 8

    eager_setup()

    # print("=== Test Encoder ===")
    # _test_encoder(batch_size=batch_size, horizon=horizon)
    #
    # print("=== Test Decoder ===")
    # _test_decoder(batch_size=batch_size, horizon=horizon)
    #
    # print("=== Test Reward Model ===")
    # _test_reward_model(batch_size=batch_size, horizon=horizon)
    #
    # print("=== Test Transition Model ===")
    # _test_transition_model(batch_size=batch_size, horizon=horizon)

    print("=== Test Training Models ===")
    _test_train(batch_size=batch_size, horizon=horizon)