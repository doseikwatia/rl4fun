from base64 import b64encode
from IPython.display import HTML
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, RecordVideo, RecordEpisodeStatistics
import gymnasium as gym
import numpy as np


def create_train_env(env_name, num_envs):
  env = gym.vector.make(env_name, num_envs=num_envs, asynchronous=False)
  env = RecordEpisodeStatistics(env)
  env = NormalizeObservation(env)
  env = NormalizeReward(env)
  return env

def create_test_env(env_name, obs_rms, video_dir, episode_trigger=lambda e: e%10==0):
  env = gym.make(env_name, render_mode= 'rgb_array')
  env = RecordVideo(env, video_dir, episode_trigger= episode_trigger )
  env = NormalizeObservation(env)
  env.obs_rms = obs_rms
  return env

def display_video(episode, video_dir):
  video_file = open(f'{video_dir}/rl-video-episode-{episode}.mp4', "r+b").read()
  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"<video width=600 controls><source src='{video_url}'></video>")


def test_agent(env, policy,video_dir, episode_trigger=lambda e: e == 0, episodes=10, max_steps=1000):
    ep_returns = []
    for ep in range(episodes):
        state,_ = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < max_steps:
            action = policy(state)
            state, reward, done, _,info = env.step(action)
            ep_ret += reward
            steps += 1
            ep_returns.append(ep_ret)

    return sum(ep_returns) / episodes  