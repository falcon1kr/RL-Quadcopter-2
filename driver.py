import numpy as np
import sys
import pandas as pd
from agents.agent import DDPG
from task import Task

def new_task(initial_pos, target_pos):
    initial_pos = initial_pos if initial_pos is not None else (np.random.rand(3) * 50) - 25
    initial_angle = np.zeros(3)

    task = Task(target_pos=target_pos, init_pose=np.concatenate((initial_pos, initial_angle)))
    task.reset()

    return initial_pos, task

def drive(num_episodes=1000, sample_distance=100, task_renew_distance=100, target_pos=np.array([0., 0., 10.]), initial_pos=None, sample_cb=None, running_size=10):
    agent = DDPG()

    positions = []
    rewards = []
    initial_poss = []
    target_poss = []
    distances = []
    times = []
    running_reward = []
    running_time = []
    running_distance = []

    max_reward = -100000

    for i_episode in range(0, num_episodes):

        if i_episode % task_renew_distance == 0:
            epi_init_pos, task = new_task(initial_pos, target_pos)
            agent.new_task(task)

        state = agent.reset_episode()

        epi_positions = []
        epi_reward = 0
        epi_distances = []

        while True:
            action = agent.act(state) 
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            epi_reward += reward

            epi_positions.append(task.sim.pose[:3])
            epi_distances.append(task.current_distance)

            if done:
                break

        avg_distance = np.average(epi_distances)

        print("\rEpisode = {:4d}, Reward = {:4n}, Avg Distance = {:4n}, time = {:4n}".format(i_episode + 1, epi_reward, avg_distance, task.sim.time), end="")

        rewards.append(epi_reward)
        distances.append(avg_distance)
        times.append(task.sim.time)

        if running_size < i_episode:
            running_reward.append(np.average(rewards[i_episode - running_size : i_episode]))
            running_time.append(np.average(times[i_episode - running_size : i_episode]))
            running_distance.append(np.average(distances[i_episode - running_size : i_episode]))
        else:
            running_reward.append(0)
            running_time.append(0)
            running_distance.append(0)

        positions.append(epi_positions)

        if i_episode % sample_distance == 0:
            max_reward = max([max_reward, epi_reward])
            initial_poss.append(epi_init_pos)
            target_poss.append(target_pos)

            if sample_cb is not None:
                sample_cb(epi_init_pos, positions, target_pos, rewards, distances, times, running_reward, running_time, running_distance)

            positions = []

        sys.stdout.flush()

    return epi_init_pos, positions, target_pos, rewards, distances, times, running_reward, running_time, running_distance

