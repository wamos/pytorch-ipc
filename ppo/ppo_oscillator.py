import gym
import gym_oscillator
import oscillator_cpp
import numpy as np
from matplotlib import pyplot as plt

# from stable_baselines.common import set_global_seeds
# from stable_baselines.common.policies import MlpPolicy,MlpLnLstmPolicy,FeedForwardPolicy
# from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize, VecEnv
# from stable_baselines import PPO2
# from stable_baselines.common.vec_env import VecEnv

from stable_baselines3 import PPO

env_id = 'oscillator-v0'
env = gym.make(env_id)
time_steps = int(1000*1000)
#time_steps = int(200000)

model = PPO("MlpPolicy", env, verbose=1)

# model.learn(time_steps)
# model.save('trained_models/trained_model_1msteps.pkl')
# quit()

model = model.load('trained_models/trained_model_1msteps.pkl')

env = gym.make(env_id)
#Store rewards
rews_ = []
#Store observations
obs_ = []
obs = env.reset()
#Store actions
acs_ = []
#Store X,Y according to 
states_x = []
states_y = []

#Initial, non-suppresssion 
for i in range(25000):
    obs, rewards, dones, info = env.step([0])
    states_x.append(env.x_val)
    states_y.append(env.y_val)
    obs_.append(obs[0])
    acs_.append(0)
    rews_.append(rewards)

#Suppression stage
for i in range(25000):
    action, _states = model.predict(obs)
   
    obs, rewards, dones, info = env.step(action)
   
    states_x.append(env.x_val)
    states_y.append(env.y_val)
    obs_.append(obs[0])
    acs_.append(action)
    rews_.append(rewards)

#Final relaxation
for i in range(5000):
    obs, rewards, dones, info = env.step([0])
    states_x.append(env.x_val)
    states_y.append(env.y_val)
    obs_.append(obs[0])
    acs_.append(0)
    rews_.append(rewards)

plt.figure(figsize=(25,5))
plt.title('Suppression plot')
plt.xlabel('TimeStep')
plt.ylabel('Signal Value')
plt.plot(states_x)
plt.savefig("suppression.png", dpi=100)