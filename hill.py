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

    for attempt in range(restart_limit):
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

        if cost < best_cost:
            best_cost = cost
            best_path = path

    return best_path, best_cost, image_paths

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
    path, cost, image_files = randomized_climb(city_locations)
    end_time = time.time()

    print("Final Tour Order:", [int(i) for i in path])
    print("Minimum Distance Achieved:", round(cost, 2))
    print("Execution Duration:", round(end_time - start_time, 4), "seconds")

    # Save GIF Animation
    frames = [imageio.imread(f) for f in image_files]
    imageio.mimsave("tsp_hillclimb_result.gif", frames, fps=2)

    # Clean up temporary files
    for f in image_files:
        os.remove(f)

    print(f"Animation saved with {len(image_files)} steps.")
