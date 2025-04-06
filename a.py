from queue import PriorityQueue
import gymnasium as gym
import time
import imageio

def branch_and_bound(env, max_time=600, gif_path="bnb_search.gif"):
    start_time = time.time()
    visited = set()
    queue = PriorityQueue()
    frames = []

    # Reset and get initial state
    state, _ = env.reset()
    frames.append(env.render())

    queue.put((0, 0, state, []))  # (priority, cost, state, path)

    while not queue.empty() and (time.time() - start_time) < max_time:
        _, cost, current_state, path = queue.get()

        if current_state in visited:
            continue
        visited.add(current_state)

        env.unwrapped.s = current_state
        frames.append(env.render())

        # Check for goal
        if env.unwrapped.desc.reshape(-1)[current_state] == b'G':
            imageio.mimsave(gif_path, frames, duration=1.5)  # Slower gif
            return path, len(path), time.time() - start_time

        # Explore next states
        for action in range(env.action_space.n):
            prob, next_state, _, _ = env.unwrapped.P[current_state][action][0]
            if prob > 0 and next_state not in visited:
                queue.put((cost + 1, cost + 1, next_state, path + [action]))

    # Save what was explored if goal not found
    imageio.mimsave(gif_path, frames, duration=1.5)
    return None, -1, time.time() - start_time


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array")
    path, steps, duration = branch_and_bound(env, gif_path="bnb_search.gif")
    print(f"Path: {path}")
    print(f"Steps: {steps}")
    print(f"Time: {duration:.2f} seconds")
