from __future__ import print_function
import gym
import gym_oscillator # for using oscillator in gym env
import oscillator_cpp # for the oscillator code
import argparse
import multiprocessing as mp
import torch
from time import sleep
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO
from matplotlib import pyplot as plt

def train(model: BaseAlgorithm, iter: int, num_steps: int):
  model.learn(total_timesteps=num_steps)
  print("train ", num_steps, " steps in iteration", iter)

def multiprcess_predict(queue: mp.Queue, iters: int):
  print("predict in ", iter)
  current_iter = 0
  env = gym.make("oscillator-v0")
  model = PPO("MlpPolicy", env, device="cpu", verbose=1)
  obs = env.reset()
  while(current_iter < iters):
    if(queue.empty() == False):
      params_dicts = queue.get()
      model.set_parameters(params_dicts)
      print("recv from training process!", current_iter, "iteration")
      current_iter = current_iter + 1
    else:
      print("queue is empty!")
      pass

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
    
    # plt.figure(figsize=(25,5))
    # plt.title('Suppression plot')
    # plt.xlabel('TimeStep')
    # plt.ylabel('Signal Value')
    # plt.plot(states_x)
    # plt.savefig("suppression.png", dpi=100)
    print("inference and plotting done at", current_iter, "iteration")
    
def main():
  parser = argparse.ArgumentParser(description='sb3 PPO mp')
  parser.add_argument('--iters', type=int, default=10, metavar='N',
                      help='number of epochs to train (default: 10)')
  args = parser.parse_args()

  env = gym.make('oscillator-v0')
  model = PPO("MlpPolicy", env, verbose=1)
  # model.learn(total_timesteps=10)
  # params_dicts = model.get_parameters()
  # for k,v in params_dicts.items():
  #   print(k)
  #   print(type(v))
  #   # for k1, v1 in v.items():
  #   #   print(v1.size())

  # del model
  # model2 = PPO("MlpPolicy", env, verbose=1)
  # model2.set_parameters(params_dicts)

  #multi-processing
  mp.set_start_method('spawn')
  ipc_queue = mp.Queue()    
  inf_device = torch.device("cpu")  
  inf_process = mp.Process(target=multiprcess_predict, args=(ipc_queue, args.iters))
  inf_started=False


  for iter in range(1, args.iters + 1):
    train(model, iter, 1000)
    if inf_started == False:
      inf_process.start()
      inf_started = True

    params_dicts = model.get_parameters()    
    ipc_queue.put(params_dicts)
    sleep(0.5)

  env.close()
  print("break from for loop")
  inf_process.join()
  print("inf_process joined")

if __name__ == '__main__':
    main()