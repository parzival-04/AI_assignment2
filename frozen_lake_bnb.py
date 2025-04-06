import gymnasium as gym
import heapq
import imageio
import numpy as np
import time
import matplotlib.pyplot as plt

def heuristic(state, env):
    goal_state = env.observation_space.n - 1
    size = int(np.sqrt(env.observation_space.n))
    current_row = state // size
    current_col = state % size
    goal_row = goal_state // size
    goal_col = goal_state % size
    return abs(goal_row - current_row) + abs(goal_col - current_col)

def branch_and_bound(env, frame_callback=None):
    start_time = time.time()
    actual_env = env.unwrapped

    start_state, _ = env.reset()
    goal_state = env.observation_space.n - 1
    best_path_length = float('inf')
    best_path = []
    visited = {}

    heap = []
    initial_h = heuristic(start_state, env)
    heapq.heappush(heap, (initial_h, 0, start_state, [start_state]))
    visited[start_state] = 0

    while heap:
        current_f, current_len, current_state, current_path = heapq.heappop(heap)

        if frame_callback:
            env.unwrapped.s = current_state
            frame_callback(env.render())

        if current_f >= best_path_length:
            continue

        if current_state == goal_state:
            if current_len < best_path_length:
                best_path_length = current_len
                best_path = current_path
            continue

        for action in range(env.action_space.n):
            transitions = actual_env.P[current_state][action]
            prob, next_state, reward, done = transitions[0]

            if done and next_state != goal_state:
                continue

            new_len = current_len + 1
            if next_state in visited and visited[next_state] <= new_len:
                continue

            new_path = current_path + [next_state]
            visited[next_state] = new_len

            h = heuristic(next_state, env)
            f = new_len + h

            if f < best_path_length:
                heapq.heappush(heap, (f, new_len, next_state, new_path))

    end_time = time.time()
    return best_path, best_path_length, end_time - start_time

def get_action(current_state, next_state, env):
    actual_env = env.unwrapped
    for action in range(env.action_space.n):
        transitions = actual_env.P[current_state][action]
        prob, s, reward, done = transitions[0]
        if s == next_state:
            return action
    return None

def generate_search_gif(best_path, search_frames, filename='full_search.gif'):
    frames = search_frames.copy()
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array')
    state, _ = env.reset()
    env.unwrapped.s = best_path[0]
    frames.append(env.render())

    for i in range(1, len(best_path)):
        env.unwrapped.s = best_path[i]
        frames.append(env.render())

    env.close()
    imageio.mimsave(filename, frames, fps=2)

def plot_avg_time(times, filename='frozen_avg_time.png'):
    avg_time = np.mean(times)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(times)+1), times, marker='o', label='Run Time (s)')
    plt.axhline(avg_time, color='r', linestyle='--', label=f'Average Time: {avg_time:.4f}s')
    plt.xlabel('Run')
    plt.ylabel('Time (seconds)')
    plt.title('Branch and Bound on FrozenLake - Time per Run')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    num_runs = 5
    times = []
    best_path_all_runs = []

    for run in range(num_runs):
        env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array')
        search_frames = []

        def frame_callback(frame):
            search_frames.append(frame)

        best_path, best_length, time_taken = branch_and_bound(env, frame_callback if run == 0 else None)
        times.append(time_taken)
        best_path_all_runs.append(best_path)
        env.close()

        if run == 0:
            print(f"Run {run+1}:")
            print(f"Best Path Length: {best_length}")
            print(f"Time Taken: {time_taken:.4f} seconds")
            print(f"Path States: {best_path}")
            generate_search_gif(best_path, search_frames)

    plot_avg_time(times)
    print(f"\nAverage Time over {num_runs} runs: {np.mean(times):.4f} seconds")
    print("Saved plot as 'frozen_avg_time.png'")
