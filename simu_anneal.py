import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import random
import time
import os

def read_city_file(filename):
    coordinates = []
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
                parts = line.split()
                x_coord, y_coord = float(parts[1]), float(parts[2])
                coordinates.append([x_coord, y_coord])
    return np.array(coordinates)

def generate_distance_table(points):
    return np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)

def route_cost(path, dist_table):
    shifted = np.roll(path, -1)
    return np.sum(dist_table[path, shifted])

def swap_cities(route):
    new_route = route.copy()
    i, j = np.random.choice(len(route), 2, replace=False)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def simulated_annealing_tsp(cities, run_no, initial_temp=1000, cooling_rate=0.995, min_temp=1e-3, max_iters=100000):
    dist_table = generate_distance_table(cities)
    current_route = list(np.random.permutation(len(cities)))
    current_cost = route_cost(current_route, dist_table)
    best_route = current_route
    best_cost = current_cost
    temp = initial_temp
    frame_idx = 0
    frame_paths = []

    for iteration in range(max_iters):
        candidate_route = swap_cities(current_route)
        candidate_cost = route_cost(candidate_route, dist_table)

        delta = candidate_cost - current_cost
        accept = delta < 0 or random.random() < np.exp(-delta / temp)

        if accept:
            current_route = candidate_route
            current_cost = candidate_cost

            if current_cost < best_cost:
                best_route = current_route
                best_cost = current_cost

                # Save frame with run and stats
                plt.figure(figsize=(7, 5))
                full_path = best_route + [best_route[0]]
                plt.scatter(cities[:, 0], cities[:, 1], c='black')
                plt.plot(cities[full_path, 0], cities[full_path, 1], c='darkorange')
                plt.title(f"Run {run_no} | Iter {iteration} | Temp: {temp:.2f} | Cost: {best_cost:.2f}")
                plt.axis('equal')
                frame_file = f"frame_{run_no}_{frame_idx}.png"
                plt.savefig(frame_file)
                frame_paths.append(frame_file)
                plt.close()
                frame_idx += 1

        temp *= cooling_rate
        if temp < min_temp:
            break

    return best_route, best_cost, frame_paths

# ---- Main Execution ----
city_locations = read_city_file("TSP.txt")

# Plot initial city map
plt.figure()
plt.scatter(city_locations[:, 0], city_locations[:, 1], color='darkgreen')
plt.title("TSP City Map")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.savefig("initial_map.png")
plt.show()

# ---- Run Multiple Times and Record Stats ----
num_runs = 5
distances = []
times = []
frame_counts = []
all_frames = []
best_overall_cost = float('inf')
best_overall_path = None
best_run = -1

for run in range(num_runs):
    print(f"\n--- Run {run+1} ---")
    start_time = time.time()
    best_path, best_cost, frame_files = simulated_annealing_tsp(city_locations, run_no=run+1)
    end_time = time.time()

    elapsed = round(end_time - start_time, 4)

    print("Optimal Path Found:", best_path)
    print("Total Distance:", best_cost)
    print("Elapsed Time:", elapsed, "seconds")

    distances.append(best_cost)
    times.append(elapsed)
    frame_counts.append(len(frame_files))

    # Save best overall
    if best_cost < best_overall_cost:
        best_overall_cost = best_cost
        best_overall_path = best_path
        best_run = run + 1

    # Collect all frames for combined gif
    for frame_file in frame_files:
        if os.path.exists(frame_file):
            all_frames.append(imageio.imread(frame_file))
            os.remove(frame_file)

# Save combined GIF
gif_name = "simulated_annealing_tsp_all_runs.gif"
imageio.mimsave(gif_name, all_frames, fps=2)
print(f"\n🎥 Combined GIF saved as {gif_name} with {len(all_frames)} frames.")

# ---- Final Best Path Plot ----
plt.figure(figsize=(8, 6))
full_best = best_overall_path + [best_overall_path[0]]
plt.scatter(city_locations[:, 0], city_locations[:, 1], color='black')
plt.plot(city_locations[full_best, 0], city_locations[full_best, 1], color='red', linewidth=2)
plt.title(f"Best Path Overall Simu Anneal | Run {best_run} | Distance: {best_overall_cost:.2f}")
plt.axis('equal')
plt.savefig("best_overall_path_simu.png")
plt.show()

# ---- Plot Stats ----
labels = [f'Run {i+1}' for i in range(num_runs)]
x = np.arange(num_runs)
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, distances, width, label='Total Distance', color='orange')
plt.bar(x, times, width, label='Time (s)', color='green')
plt.bar(x + width, frame_counts, width, label='Frames', color='blue')

plt.xlabel('Run')
plt.ylabel('Metrics')
plt.title('Simulated Annealing TSP Stats per Run')
plt.xticks(x, labels)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("simu_anneal_stats.png")
plt.show()
