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

def simulated_annealing_tsp(cities, initial_temp=1000, cooling_rate=0.995, min_temp=1e-3, max_iters=100000):
    dist_table = generate_distance_table(cities)
    current_route = list(np.random.permutation(len(cities)))
    current_cost = route_cost(current_route, dist_table)
    best_route = current_route
    best_cost = current_cost
    temp = initial_temp
    frame_idx = 0

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

                # Save frame
                plt.figure(figsize=(7, 5))
                full_path = best_route + [best_route[0]]
                plt.scatter(cities[:, 0], cities[:, 1], c='black')
                plt.plot(cities[full_path, 0], cities[full_path, 1], c='darkorange')
                plt.title(f"Iteration {iteration} | Temp: {temp:.2f}")
                plt.axis('equal')
                plt.savefig(f"snapshot_{frame_idx}.png")
                plt.close()
                frame_idx += 1

        temp *= cooling_rate
        if temp < min_temp:
            break

    return best_route, best_cost, frame_idx

# ---- Main Execution ----
city_locations = read_city_file("TSP.txt")

# Plot city map
plt.figure()
plt.scatter(city_locations[:, 0], city_locations[:, 1], color='darkgreen')
plt.title("TSP City Map")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.savefig("initial_map.png")
plt.show()

start_time = time.time()
best_path, best_cost, frames = simulated_annealing_tsp(city_locations)
end_time = time.time()

print("Optimal Path Found:", best_path)
print("Total Distance:", best_cost)
print("Elapsed Time:", round(end_time - start_time, 4), "seconds")

# Create GIF
images = []
for idx in range(frames):
    filename = f"snapshot_{idx}.png"
    if os.path.exists(filename):
        images.append(imageio.imread(filename))
        os.remove(filename)

imageio.mimsave("simulated_annealing_tsp.gif", images, fps=1)
print(f"GIF saved with {frames} frames.")
