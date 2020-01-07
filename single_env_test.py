import itertools
from dm_control2gym.util import make_dm2gym_env_obs


def play_ground(env_name="cheetah_run"):
    env = make_dm2gym_env_obs(env_name=env_name, num_repeat_action=4)

    state = env.reset()
    print("State shape: ", state.shape)

    total_reward = 0

    for t in itertools.count():
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done: break

    env.close()
    print("Total Reward: {}".format(total_reward))


if __name__ == '__main__':
    env_name = "cartpole_balance"
    play_ground(env_name)
