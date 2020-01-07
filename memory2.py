import os
import random
import shutil
import numpy as np
from tqdm import tqdm


def check_path(path):
    if path[-1] == "/":
        return path[:-1]
    else:
        return path


class ReplayBuffer(object):
    """ Experience Replay Memory Buffer """

    def __init__(self,
                 size,
                 n_step=0,
                 gamma=0.99,
                 traj_dir="./tmp"):

        self._o_table = list()  # observation at t
        self._a_table = list()  # action at t
        self._r_table = list()  # reward at t
        self._no_table = list()  # next_observation at t
        self._d_table = list()  # done at t

        self._maxsize = size
        self._n_step = n_step
        self._gamma = gamma
        self._next_idx = 0
        self._saved_idx = 0
        self._epoch = 0
        self.traj_dir = check_path(path=traj_dir)

        # load previous trajectories
        self.load()

    @property
    def epoch(self):
        return self._epoch

    def __len__(self):
        return len(self._a_table)

    def add(self, obs_t, action, reward, obs_tp1, done):
        """ Register data to the tables """
        if self._next_idx >= len(self._a_table):
            self._o_table.append(obs_t)
            self._a_table.append(action)
            self._r_table.append(reward)
            self._no_table.append(obs_tp1)
            self._d_table.append(done)
        else:
            self._o_table[self._next_idx] = obs_t
            self._a_table[self._next_idx] = action
            self._r_table[self._next_idx] = reward
            self._no_table[self._next_idx] = obs_tp1
            self._d_table[self._next_idx] = done
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        """
        One step sampling method
        """
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            obses_t.append(np.array(self._o_table[i], copy=False))
            actions.append(np.array(self._a_table[i], copy=False))
            rewards.append(self._r_table[i])
            obses_tp1.append(np.array(self._no_table[i], copy=False))
            dones.append(self._d_table[i])
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _encode_sample_n_step_sequence(self, idxes):
        """
        n-consecutive time-step sampling method

        :return: obs, act, rew, next_obs, done FROM t to t+n
        """
        # Resulting arrays
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []

        # === Sampling method ===
        for i in idxes:
            if i + self._n_step > len(self._a_table) - 1:  # avoid the index out of range error!!
                first_half = len(self._a_table) - 1 - i
                second_half = self._n_step - first_half
                o_seq = self._o_table[i: i + first_half] + self._o_table[:second_half]
                a_seq = self._a_table[i: i + first_half] + self._a_table[:second_half]
                r_seq = self._r_table[i: i + first_half] + self._r_table[:second_half]
                no_seq = self._no_table[i: i + first_half] + self._no_table[:second_half]
                d_seq = self._d_table[i: i + first_half] + self._d_table[:second_half]
            else:
                o_seq = self._o_table[i: i + self._n_step]
                a_seq = self._a_table[i: i + self._n_step]
                r_seq = self._r_table[i: i + self._n_step]
                no_seq = self._no_table[i: i + self._n_step]
                d_seq = self._d_table[i: i + self._n_step]

            o_seq, a_seq, r_seq, no_seq, d_seq = self._check_episode_end(o_seq, a_seq, r_seq, no_seq, d_seq)

            # Store a data at each time-sequence in the resulting array
            obses_t.append(np.array(o_seq, copy=False))
            actions.append(a_seq)
            rewards.append(r_seq)
            obses_tp1.append(np.array(no_seq, copy=False))
            dones.append(d_seq)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _check_episode_end(self, o_seq, a_seq, r_seq, no_seq, d_seq):
        """ validate if the extracted part of the memory is semantically correct.
        """

        """ Why this works?
            Note that `np.argmax` returns the first index of the largest numbers in an array.
            So that, it can cover all possible combinations of done_flg in d_seq.
        """
        _max = np.max(np.asarray(d_seq).astype(np.float32)).astype(np.int32)
        _id = np.argmax(np.asarray(d_seq).astype(np.float32)).astype(np.int32)

        if _max == 1:
            # if it has the terminal state in an episode in the extracted sequence
            # Except done_flg... we don't want to change the semantics of done_flg
            o_seq = self._zero_padding(_id, o_seq)
            a_seq = self._zero_padding(_id, a_seq)
            no_seq = self._zero_padding(_id, no_seq)
            r_seq = self._zero_padding(_id, r_seq)
            return o_seq, a_seq, r_seq, no_seq, d_seq
        else:
            # if it doesn't have the terminal state in an episode in the extracted sequence
            return o_seq, a_seq, r_seq, no_seq, d_seq

    def _zero_padding(self, index, data):
        """ Pad the states after termination by 0s """
        _sample = np.asarray(data[0])
        _dtype, _shape = _sample.dtype, _sample.shape
        before_terminate = data[:index]
        after_terminate = np.zeros(shape=(self._n_step - index,) + _shape, dtype=_dtype).tolist()
        return np.asarray(before_terminate + after_terminate)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._a_table) - 1) for _ in range(batch_size)]
        if self._n_step == 0:
            obses_t, actions, rewards, obses_tp1, dones = self._encode_sample(idxes)
        else:
            obses_t, actions, rewards, obses_tp1, dones = self._encode_sample_n_step_sequence(idxes)
        obses_t, actions, rewards, obses_tp1, dones = self._after_sampling(obses_t, actions, rewards, obses_tp1, dones)
        return obses_t, actions, rewards, obses_tp1, dones

    def _after_sampling(self, obses_t, actions, rewards, obses_tp1, dones):
        """ Subclass implements here to normalise or do some pre-processing for the training """
        return obses_t, actions, rewards, obses_tp1, dones

    def _get_save_idx(self):
        if self._saved_idx < self._next_idx:
            idxes = np.arange(start=self._saved_idx, stop=self._next_idx).tolist()
        else:
            first_half = np.arange(start=self._saved_idx, stop=self._maxsize).tolist()
            second_half = np.arange(start=0, stop=self._next_idx).tolist()
            idxes = first_half + second_half
        return idxes

    def save(self, epoch=0):
        """ save tables """
        if not os.path.isdir(self.traj_dir + "/{}".format(epoch)):
            os.makedirs(self.traj_dir + "/{}".format(epoch))
        else:
            shutil.rmtree(self.traj_dir + "/{}".format(epoch))
            os.makedirs(self.traj_dir + "/{}".format(epoch))

        idxes = self._get_save_idx()
        obses_t, actions, rewards, obses_tp1, dones = self._encode_sample(idxes=idxes)

        np.save(self.traj_dir + "/{}/next_idx".format(epoch), self._next_idx)
        np.save(self.traj_dir + "/{}/o".format(epoch), obses_t)
        np.save(self.traj_dir + "/{}/a".format(epoch), actions)
        np.save(self.traj_dir + "/{}/no".format(epoch), obses_tp1)
        np.save(self.traj_dir + "/{}/r".format(epoch), rewards)
        np.save(self.traj_dir + "/{}/d".format(epoch), dones)

        # update the saved index
        self._saved_idx = self._next_idx

    def load(self):
        """ load tables """

        if os.path.isdir(self.traj_dir):
            print("\n<< ========| Loading Previous Trajectories |======== >>")
            num_dirs = int(np.max(np.array([int(i) for i in os.listdir(self.traj_dir)])))
            begin_epoch = int(np.min(np.array([int(i) for i in os.listdir(self.traj_dir)])))
            for epoch in tqdm(range(begin_epoch, num_dirs + 1)):
                if epoch == begin_epoch:
                    self._next_idx = int(np.load(self.traj_dir + "/{}/next_idx.npy".format(epoch)))
                    self._o_table = np.load(self.traj_dir + "/{}/o.npy".format(epoch))
                    self._a_table = np.load(self.traj_dir + "/{}/a.npy".format(epoch))
                    self._no_table = np.load(self.traj_dir + "/{}/no.npy".format(epoch))
                    self._r_table = np.load(self.traj_dir + "/{}/r.npy".format(epoch))
                    self._d_table = np.load(self.traj_dir + "/{}/d.npy".format(epoch))
                else:
                    self._next_idx = int(np.load(self.traj_dir + "/{}/next_idx.npy".format(epoch)))
                    self._o_table = np.concatenate([self._o_table,
                                                    np.load(self.traj_dir + "/{}/o.npy".format(epoch))], axis=0)
                    self._a_table = np.concatenate([self._a_table,
                                                    np.load(self.traj_dir + "/{}/a.npy".format(epoch))], axis=0)
                    self._no_table = np.concatenate([self._no_table,
                                                     np.load(self.traj_dir + "/{}/no.npy".format(epoch))], axis=0)
                    self._r_table = np.concatenate([self._r_table,
                                                    np.load(self.traj_dir + "/{}/r.npy".format(epoch))], axis=0)
                    self._d_table = np.concatenate([self._d_table,
                                                    np.load(self.traj_dir + "/{}/d.npy".format(epoch))], axis=0)

            self._o_table = list(self._o_table)  # observation at t
            self._a_table = list(self._a_table)  # action at t
            self._r_table = list(self._r_table)  # reward at t
            self._no_table = list(self._no_table)  # next_observation at t
            self._d_table = self._d_table.tolist()  # done at t
            self._epoch = epoch  # set the latest epoch

            print("\n===================================================\n")
            print(" Finish Loading previous trajectories")
            print(" Path: {}".format(self.traj_dir))
            print("\n===================================================\n")

        else:
            print("\n===================================================\n")
            print(" Previous trajectories are not found")
            print(" Path: {}".format(self.traj_dir))
            print("\n===================================================\n")

    def refresh(self):
        self._o_table = list()  # observation at t
        self._a_table = list()  # action at t
        self._r_table = list()  # reward at t
        self._no_table = list()  # next_observation at t
        self._d_table = list()  # done at t
        self._next_idx = 0
