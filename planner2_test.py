from planner2 import MPCPlanner


def play_ground(model, env):
    total_reward = 0
    state = env.reset()
    for t in itertools.count():
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done: break

    env.close()
    print("Total Reward: {}".format(total_reward))


if __name__ == '__main__':
    import itertools
    from dm_control2gym.util import make_dm2gym_env_obs

    mpc = MPCPlanner(transition_model=None, reward_model=None)
    env_name = "cartpole_balance"
    env = make_dm2gym_env_obs(env_name=env_name, num_repeat_action=4)
    print("State shape: {}, Action dim: {}".format(env.observation_space.shape, env.action_space.shape))
    play_ground(model=mpc, env=env)
