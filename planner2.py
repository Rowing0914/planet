import tensorflow as tf


class MPCPlanner(object):
    def __init__(self,
                 transition_model,
                 reward_model,
                 action_size=1,
                 planning_horizon=12,
                 optimisation_iters=5,
                 candidates=100,
                 top_candidates=10):
        self.transition_model, self.reward_model = transition_model, reward_model
        self.action_size = action_size
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates

    def forward(self, belief, state):
        action_mean = tf.zeros(shape=(self.planning_horizon, belief.shape[0], 1, self.action_size))
        action_std = tf.ones(shape=(self.planning_horizon, belief.shape[0], 1, self.action_size))

        for iter in range(self.optimisation_iters):
            # sample a sequence of actions
            actions = (action_mean + action_std * tf.random.normal(shape=(self.planning_horizon,
                                                                          belief.shape[0],
                                                                          self.candidates,
                                                                          self.action_size)))

            # sample candidate trajectories following the sampled actions from the given state/belief
            beliefs, states, _, _ = self.transition_model(state, actions, belief)

            # aggregate the collected returns of trajectories
            returns = tf.reduce_sum(self.reward_model(belief, states), axis=0)

            # Re-fit belief to the K best action sequences
            topk = tf.math.top_k(input=returns, k=self.top_candidates)
            best_actions = actions[:, topk]

            # update belief with new mean and std
            action_mean = tf.math.reduce_mean(best_actions, keepdims=True)
            action_std = tf.math.reduce_std(best_actions, keepdims=True)
            return action_mean[0]






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
