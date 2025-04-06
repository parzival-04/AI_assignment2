import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
import os

# ---------- Utility Functions ----------

def read_city_file(filename):
    coords = []
    with open(filename, 'r') as f:
        parsing = False
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                parsing = True
                continue
            if line == "EOF":
                break
            if parsing:
                _, x, y = line.split()
                coords.append([float(x), float(y)])
    return np.array(coords)

def generate_distance_table(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def route_cost(path, dist_table):
    return sum(dist_table[path[i], path[(i+1)%len(path)]] for i in range(len(path)))

def two_opt_swap(route, dist_table):
    min_cost = route_cost(route, dist_table)
    best = route[:]
    improved = True

    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                if j == len(route) - 1 and i == 0:
                    continue  # Avoid breaking the loop
                new_route = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                new_cost = route_cost(new_route, dist_table)
                if new_cost < min_cost:
                    best = new_route
                    min_cost = new_cost
                    improved = True
                    break  # Accept first improvement
            if improved:
                break
    return best, min_cost

def plot_path(points, path, attempt, step, counter):
    plt.clf()
    plt.scatter(points[:, 0], points[:, 1], c='black')
    full_path = path + [path[0]]
    plt.plot(points[full_path, 0], points[full_path, 1], c='red')
    plt.title(f"Try {attempt+1}, Step {step+1}")
    plt.axis('equal')
    fname = f"snapshot_{counter}.png"
    plt.savefig(fname)
    return fname

def randomized_climb(points, restart_limit=5, iteration_limit=1000, base_seed=101):
    dist_table = generate_distance_table(points)
    best_path = None
    best_cost = float('inf')
    image_paths = []
    best_costs_per_run = []
    time_taken_per_run = []

    for attempt in range(restart_limit):
        start = time.time()
        rng = np.random.default_rng(base_seed + attempt)
        path = list(rng.permutation(len(points)))
        cost = route_cost(path, dist_table)

        for step in range(iteration_limit):
            new_path, new_cost = two_opt_swap(path, dist_table)
            if new_cost < cost:
                path = new_path
                cost = new_cost
                img = plot_path(points, path, attempt, step, len(image_paths))
                image_paths.append(img)
            else:
                break

        time_taken = time.time() - start
        best_costs_per_run.append(cost)
        time_taken_per_run.append(time_taken)

        if cost < best_cost:
            best_cost = cost
            best_path = path

    return best_path, best_cost, image_paths, best_costs_per_run, time_taken_per_run

# ---------- Main Execution ----------

if __name__ == "__main__":
    city_locations = read_city_file("TSP.txt")

    # Plot Initial City Map
    plt.figure(figsize=(7, 5))
    plt.scatter(city_locations[:, 0], city_locations[:, 1], color='black')
    plt.title("Map of Cities")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.savefig("initial_map.png")
    plt.close()

    # Run Randomized Hill Climbing
    start_time = time.time()
    path, cost, image_files, best_costs, times = randomized_climb(city_locations)
    end_time = time.time()

    print("Final Tour Order:", [int(i) for i in path])
    print("Minimum Distance Achieved:", round(cost, 2))
    print("Total Execution Duration:", round(end_time - start_time, 4), "seconds")

    # Save GIF Animation
    frames = [imageio.imread(f) for f in image_files]
    imageio.mimsave("tsp_hillclimb_result.gif", frames, fps=2)

    # Clean up temporary files
    for f in image_files:
        os.remove(f)

    print(f"Animation saved with {len(image_files)} steps.")

    # Plot Combined Stats: Distance and Time
    x = np.arange(len(best_costs))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars1 = ax1.bar(x - width/2, best_costs, width, label='Best Distance', color='steelblue')
    ax1.set_ylabel('Distance')
    ax1.set_xlabel('Run #')
    ax1.set_title('Hill Climb Stats per Run')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Run {i+1}' for i in x])
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, times, width, label='Time Taken (s)', color='darkorange')
    ax2.set_ylabel('Time (s)')
    ax2.tick_params(axis='y')

    # Combine Legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    fig.tight_layout()
    plt.savefig("hill_climb_stats.png")
    plt.show()
