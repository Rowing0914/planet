import numpy as np
import tensorflow as tf


class Controller(object):
    def __init__(self, planner, encoder, belief_size=200, state_size=30, action_size=1, explore=True, action_noise=0.1):
        self._belief_size = belief_size
        self._state_size = state_size
        self._action_size = action_size
        self._explore = explore
        self._action_noise = action_noise
        self.planner = planner
        self.encoder = encoder

        self._prep_controller()

    def prep_controller(self):
        self._prep_controller()

    def _prep_controller(self):
        self._belief = np.zeros(shape=(1, self._belief_size)).astype(np.float32)
        self._posterior_state = np.zeros(shape=(1, self._state_size)).astype(np.float32)
        self._action = np.zeros(shape=(1, self._action_size)).astype(np.float32)

    def select_action(self, obs):
        action = self._select_action(obs).numpy()

        if self._explore:
            action += self._action_noise * np.random.randn(*action.shape)

        return action

    @tf.contrib.eager.defun
    def _select_action(self, obs):
        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        belief, _, _, _, posterior_state, _, _ = self.planner.transition_model(prev_state=self._posterior_state,
                                                                               actions=self._action[np.newaxis, ...],
                                                                               prev_belief=self._belief,
                                                                               obs=self.encoder(obs)[np.newaxis, ...])

        # remove the horizon dimension
        belief, posterior_state = tf.squeeze(belief, axis=0), tf.squeeze(posterior_state, axis=0)

        # forward planning to get action from planner(q(s_t|o≤t,a<t), p)
        action = self.planner.forward(belief=belief, state=posterior_state)

        return action


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
        """ Forward Planning algorithm

        :param belief: batch x belief_size
        :param state: batch x belief_size
        """
        # get shapes
        batch_size, belief_size, state_size = tf.shape(belief)[0], tf.shape(belief)[1], tf.shape(state)[1]

        # reshape components
        belief = tf.tile(tf.expand_dims(belief, axis=1), multiples=(batch_size, self.candidates, 1))
        belief = tf.reshape(belief, shape=(-1, belief_size))
        state = tf.tile(tf.expand_dims(state, axis=1), multiples=(batch_size, self.candidates, 1))
        state = tf.reshape(state, shape=(-1, state_size))

        # initialise the params of the action dist(mean/std)
        action_mean = tf.zeros(shape=(self.planning_horizon, batch_size, 1, self.action_size))
        action_std = tf.ones(shape=(self.planning_horizon, batch_size, 1, self.action_size))

        # optimise the action dist
        for iter in range(self.optimisation_iters):
            # sample a sequence of actions
            noise = tf.random.normal(shape=(self.planning_horizon, batch_size, self.candidates, self.action_size))
            actions = (action_mean + action_std * noise)  # horizon x batch x candidates x action_size
            actions = tf.reshape(actions, shape=(self.planning_horizon, batch_size * self.candidates, self.action_size))

            # sample candidate trajectories following the sampled actions from the given state/belief
            _belief, _state, _, _ = self.transition_model(state, actions, belief)  # horizon x candidate x dim

            # merge the time-horizon axis to the batch-size axis
            _belief, _state = tf.reshape(_belief, shape=(-1, belief_size)), tf.reshape(_state, shape=(-1, state_size))

            # aggregate the collected returns of trajectories
            returns = self.reward_model(_belief, _state)  # (horizon*candidate) x 1
            returns = tf.reshape(returns, shape=(self.planning_horizon, -1))  # horizon x candidate
            returns = tf.reduce_sum(returns, axis=0)  # (candidate, )

            # Re-fit belief to the K best action sequences
            _, topk_id = tf.math.top_k(input=returns, k=self.top_candidates)  # (top_candidate,)
            best_actions = tf.gather(actions, topk_id)  # top_candidate x action_size

            # update belief with new mean and std
            action_mean = tf.math.reduce_mean(best_actions, keepdims=True)
            action_std = tf.math.reduce_std(best_actions, keepdims=True)
        return action_mean[0]
