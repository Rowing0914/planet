import time
import numpy as np
import tensorflow as tf

from planner2 import MPCPlanner, Controller
from models2 import TransitionModel, RewardModel, Encoder
from eager_setup import eager_setup

eager_setup()


def play_ground(planner, env, encoder, belief_size=200, state_size=30, action_noise=0.1, explore=True):
    total_reward = 0
    obs = env.reset()

    belief = np.zeros(shape=(1, belief_size)).astype(np.float32)
    posterior_state = np.zeros(shape=(1, state_size)).astype(np.float32)
    action = np.zeros(shape=(1, env.action_space.shape[0])).astype(np.float32)

    for t in itertools.count():
        # preprocess obs
        obs = (obs / 255.0).astype(np.float32)

        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        belief, _, _, _, posterior_state, _, _ = planner.transition_model(prev_state=posterior_state,
                                                                          actions=action[np.newaxis, ...],
                                                                          prev_belief=belief,
                                                                          observations=encoder(obs)[np.newaxis, ...])

        # remove the horizon dimension
        belief, posterior_state = tf.squeeze(belief, axis=0), tf.squeeze(posterior_state, axis=0)

        # forward planning to get action from planner(q(s_t|o≤t,a<t), p)
        action = planner.forward(belief=belief, state=posterior_state)

        action = action.numpy()

        if explore:
            action += action_noise * np.random.randn(*action.shape)

        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs

        if done: break

    env.close()
    print("Total Reward: {}".format(total_reward))


def play_ground2(controller, env):
    total_reward = 0
    obs = env.reset()

    controller.prep_controller()

    for t in itertools.count():
        # preprocess obs
        obs = (obs / 255.0).astype(np.float32)
        action = controller.select_action(obs=obs)
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs

        if done: break

    env.close()
    print("Total Reward: {}".format(total_reward))


def _test_1():
    mpc = MPCPlanner(transition_model=TransitionModel(), reward_model=RewardModel())
    encoder = Encoder()
    env = make_dm2gym_env_obs(env_name="cartpole_balance", num_repeat_action=4)

    print("State shape: {}, Action dim: {}".format(env.observation_space.shape, env.action_space.shape))
    play_ground(planner=mpc, env=env, encoder=encoder)


def _test_2():
    env = make_dm2gym_env_obs(env_name="cartpole_balance", num_repeat_action=4)
    planner = MPCPlanner(transition_model=TransitionModel(), reward_model=RewardModel())
    encoder = Encoder()

    controller = Controller(planner=planner,
                            encoder=encoder,
                            belief_size=200,
                            state_size=30,
                            action_size=env.action_space.shape[0],
                            action_noise=0.1,
                            explore=True)

    begin = time.time()
    play_ground2(controller=controller, env=env)
    print("it took: {:.4f}[s] for one episode".format(time.time() - begin))


if __name__ == '__main__':
    import itertools
    from dm_control2gym.util import make_dm2gym_env_obs

    # print("=== Test 1 ===")
    # _test_1()

    print("=== Test 2 ===")
    _test_2()