"""
unit test for new memory buffer over n-steps
"""
import time
from dm_control2gym.util import make_dm2gym_env_obs
from memory2 import ReplayBuffer

BUFFER_SIZE = 1000

env = make_dm2gym_env_obs(env_name="cartpole_balance")
memory = ReplayBuffer(size=BUFFER_SIZE, n_step=8)

# collect data
state = env.reset()
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)
for _ in range(BUFFER_SIZE): memory.add(state, action, reward, next_state, done)
print("After data collection, Memory length: ", len(memory))

# mini-batch sampling test
obses_t, actions, rewards, obses_tp1, dones = memory.sample(batch_size=32)
print(obses_t.shape)

# check exec speed
begin = time.time()
for _ in range(1000): memory.sample(batch_size=32)
print("[sample] Took : {:3f}s".format(time.time() - begin))

# save and load test
for epoch in range(5):
    for _ in range(BUFFER_SIZE // 5): memory.add(state, action, reward, next_state, done)

    begin = time.time()
    memory.save(epoch=epoch)
    print("Epoch: {}, [save] Took : {:3f}s".format(epoch, time.time() - begin))

# load test
del memory
begin = time.time()
memory = ReplayBuffer(size=BUFFER_SIZE, n_step=8)
print("[load] Took : {:3f}s".format(time.time() - begin))
print("Memory length: ", len(memory))

# try loading wrong dir
del memory
ReplayBuffer(size=BUFFER_SIZE, traj_dir="nan")
