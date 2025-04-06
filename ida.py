import gymnasium as gym
import imageio
import numpy as np
import matplotlib.pyplot as plt
import time

def heuristic(state, env):
    goal_state = env.observation_space.n - 1
    size = int(np.sqrt(env.observation_space.n))
    current_row = state // size
    current_col = state % size
    goal_row = goal_state // size
    goal_col = goal_state % size
    return abs(goal_row - current_row) + abs(goal_col - current_col)

def ida_star(env, frame_callback=None):
    start_time = time.time()
    start_state, _ = env.reset()
    goal_state = env.observation_space.n - 1

    threshold = heuristic(start_state, env)
    path = [start_state]
    visited = set()

    while True:
        temp = search(env, path, 0, threshold, goal_state, visited, frame_callback)
        if isinstance(temp, list):  # Found path
            end_time = time.time()
            return temp, len(temp) - 1, end_time - start_time
        if temp == float('inf'):
            return None, float('inf'), time.time() - start_time  # No path found
        threshold = temp

def search(env, path, g, threshold, goal_state, visited, frame_callback):
    node = path[-1]
    f = g + heuristic(node, env)

    if frame_callback:
        env.unwrapped.s = node
        frame_callback(env.render())

    if f > threshold:
        return f
    if node == goal_state:
        return path

    min_threshold = float('inf')
    visited.add(node)

    for action in range(env.action_space.n):
        transitions = env.unwrapped.P[node][action]
        prob, next_state, reward, done = transitions[0]

        if done and next_state != goal_state:
            continue
        if next_state in path:
            continue

        path.append(next_state)
        temp = search(env, path, g + 1, threshold, goal_state, visited, frame_callback)
        if isinstance(temp, list):
            return temp  # Found goal
        if temp < min_threshold:
            min_threshold = temp
        path.pop()

    visited.remove(node)
    return min_threshold

def generate_search_gif(best_path, search_frames, filename='ida_star_search.gif'):
    frames = search_frames.copy()

    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array')
    env.reset()
    for state in best_path:
        env.unwrapped.s = state
        frames.append(env.render())
    env.close()

    imageio.mimsave(filename, frames, fps=2)

def plot_avg_time(times, filename='ida_avg_time.png'):
    avg_time = np.mean(times)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(times)+1), times, marker='o', label='Run Time (s)')
    plt.axhline(avg_time, color='red', linestyle='--', label=f'Avg Time: {avg_time:.4f}s')
    plt.xlabel('Run')
    plt.ylabel('Time (seconds)')
    plt.title('IDA* on FrozenLake - Time per Run')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    num_runs = 5
    times = []

    for run in range(num_runs):
        env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array')
        search_frames = []

        def frame_callback(frame):
            search_frames.append(frame)

        best_path, best_length, time_taken = ida_star(env, frame_callback if run == 0 else None)
        times.append(time_taken)

        if run == 0 and best_path:
            print(f"Run {run+1}:")
            print(f"Best Path Length: {best_length}")
            print(f"Time Taken: {time_taken:.4f} seconds")
            print(f"Path States: {best_path}")
            generate_search_gif(best_path, search_frames)

        env.close()

    plot_avg_time(times)
    print(f"\nAverage Time over {num_runs} runs: {np.mean(times):.4f} seconds")
    print("Saved plot as 'ida_avg_time.png'")
