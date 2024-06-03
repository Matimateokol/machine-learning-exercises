"""
   Author: Mateusz Kołacz
"""
from typing import Any
import gymnasium as gym
import numpy as np
from gymnasium import Env
from ply.cpp import xrange

class CustomRewardWrapper(gym.Wrapper):
   def __init__(self, env, custom_rewards):
      super(CustomRewardWrapper, self).__init__(env)
      self.custom_rewards = custom_rewards

   def step(self, action):
      state, reward, done, truncated, info = self.env.step(action)
      if state in self.custom_rewards:
         reward = self.custom_rewards[state]
      return state, reward, done, truncated, info

def create_custom_rewards(desc, g=1.0, h=0.0, f=0.0):
   custom_rewards = {}
   for i in range(desc.shape[0]):
      for j in range(desc.shape[1]):
         state = i * desc.shape[1] + j
         if desc[i, j] == b'G':
            custom_rewards[state] = g
         elif desc[i, j] == b'H':
            custom_rewards[state] = h
         elif desc[i, j] == b'F':
            custom_rewards[state] = f
   return custom_rewards

"""
   Custom argmax method
"""
# def argmax(arr):
#    arr_max = np.max(arr)
#    return np.random.choice(np.where(arr == arr_max)[0])

"""
   Method for choosing next action based on environment state and epsilon values
"""
def make_action(environment: Env, epsilon: float, qtable: np.ndarray[Any, np.dtype[np.float_]], state: tuple):
    # Exploration-exploitation trade-off
    if np.random.random() < epsilon:
        return environment.action_space.sample()  # Explore action space
    else:
        return np.argmax(qtable[state])  # Exploit learned values

def run_base_qlearning():
   # STEP 1 PREPARATION
   env = gym.make('FrozenLake-v1', render_mode="ansi", desc=None, map_name="8x8", is_slippery=False)
   custom_rewards = create_custom_rewards(env.unwrapped.desc)
   env = CustomRewardWrapper(env, custom_rewards)

   state_size = env.observation_space.n
   action_size = env.action_space.n


   # STEP 2 DOING THE ALGORITHM
   num_of_ind_runs = 10
   num_episodes = 1000
   averaged_reward = np.zeros(num_episodes)

   for run in xrange(num_of_ind_runs):
       print(f'Run {run+1} of {num_of_ind_runs}')
       qtable = np.zeros((state_size, action_size))

       # HYPERPARAMETERS:
       tmax = 200  # allowed steps per episode
       beta = 0.9  # learning rate
       gamma = 0.7  # discount rate
       epsilon = 0.1
       # epsilon_decay = 0.001 # (2 * epsilon) / num_episodes

       for episode in xrange(num_episodes):
           print(f"[RUN {run+1}/{num_of_ind_runs}] Episode: {episode+1} of {num_episodes}")
           state = env.reset()[0]
           # DEBUG:
           # print(env.render())
           total_rewards = 0
           done = False

           while not done:
               action_chosen = make_action(env, epsilon, qtable, state)

               # Observe the action's outcome
               next_state, reward, done, info, _ = env.step(action_chosen)

               # DEBUG:
               # print(f"Reward: {reward}")
               # print(f"Action: {action_chosen}")

               # Update Q-table
               qtable[state, action_chosen] = qtable[state, action_chosen] + beta * (reward + gamma * np.max(qtable[next_state]) - qtable[state, action_chosen])

               # DEBUG:
               # print(qtable[state, action_chosen])

               state = next_state
               total_rewards += reward

           # Decay epsilon
           # epsilon = max(epsilon - epsilon_decay, 0)

           averaged_reward[episode] = total_rewards

   averaged_reward /= num_of_ind_runs
   averaged_reward_base = averaged_reward  # niech to będą wyniki bazowe, z którymi będziemy porównywać wyniki dla innych ustawień, czy funkcji oceny
   # DEBUG:
   # print(averaged_reward_base)
   return averaged_reward_base

def run_custom_qlearning():
   # STEP 1 PREPARATION
   env = gym.make('FrozenLake-v1', render_mode="ansi", desc=None, map_name="8x8", is_slippery=False)
   custom_rewards = create_custom_rewards(env.unwrapped.desc, 100.0, -1.0, 0)
   env = CustomRewardWrapper(env, custom_rewards)

   state_size = env.observation_space.n
   action_size = env.action_space.n

   # STEP 2 DOING THE ALGORITHM
   num_of_ind_runs = 10
   num_episodes = 1000
   averaged_reward = np.zeros(num_episodes)

   for run in xrange(num_of_ind_runs):
      print(f'Run {run + 1} of {num_of_ind_runs}')
      qtable = np.zeros((state_size, action_size))

      # HYPERPARAMETERS:
      tmax = 200  # allowed steps per episode
      beta = 0.9  # learning rate
      gamma = 0.7  # discount rate
      epsilon = 0.1
      # epsilon_decay = 0.001  # (2 * epsilon) / num_episodes

      for episode in xrange(num_episodes):
         print(f"[RUN {run + 1}/{num_of_ind_runs}] Episode: {episode + 1} of {num_episodes}")
         state = env.reset()[0]
         # DEBUG:
         # print(env.render())
         total_rewards = 0
         done = False

         while not done:
            action_chosen = make_action(env, epsilon, qtable, state)

            # Observe the action's outcome
            next_state, reward, done, info, _ = env.step(action_chosen)

            # DEBUG:
            # print(f"Reward: {reward}")
            # print(f"Action: {action_chosen}")

            # Update Q-table
            qtable[state, action_chosen] = qtable[state, action_chosen] + beta * (
                       reward + gamma * np.max(qtable[next_state]) - qtable[state, action_chosen])

            # DEBUG:
            # print(qtable[state, action_chosen])

            state = next_state
            total_rewards += reward

         # Decay epsilon
         # epsilon = max(epsilon - epsilon_decay, 0)

         averaged_reward[episode] = total_rewards

   averaged_reward /= num_of_ind_runs
   # DEBUG:
   # print(averaged_reward_base)
   return averaged_reward

averaged_reward_base = run_base_qlearning()
averaged_reward_custom = run_custom_qlearning()

# STEP 3 VISUALIZATION
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ax.set_ylim(0, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(averaged_reward_base, 'r', label='Bazowa srednia nagroda')
plt.plot(averaged_reward_custom, 'b', label='Niebazowa srednia nagroda')
plt.legend()
plt.show()
