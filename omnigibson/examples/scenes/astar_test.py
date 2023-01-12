from omnigibson.maps.traversable_map import TraversableMap
import omnigibson as og
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

trav_map = TraversableMap()
scene_model = "Benevolence_2_int"
scene_dir = f"{og.og_dataset_path}/scenes/{scene_model}"
maps_path = os.path.join(scene_dir, "layout")
trav_map.load_trav_map(maps_path)
trav_map.build_graph = True

map_image = trav_map.floor_map[0]

# obstacle = []
# for i in range(map_image.shape[0]):
#     # skip obstacles outside of building
#     if len(list(np.unique(map_image[i]))) == 1:
#         continue

#     for j in range(map_image.shape[1]):
#         if map_image[i, j] == 0:
#             obstacle.append([i, j])

#             if j > 150 and len(list(np.unique(map_image[i,j-10:j+1]))) == 1:
#                 break

# print(len(obstacle))
# np.savetxt("./obstacle.csv", obstacle, delimiter=",")

a  = trav_map.map_to_world(np.array([88, 90]))
b  = trav_map.map_to_world(np.array([62, 95]))

shortest_path = np.array(
    trav_map.get_shortest_path(
        0, a, b, True
    )[0]
)

shortest_path_world = trav_map.world_to_map(shortest_path)
plt.scatter(shortest_path_world[:, 1], shortest_path_world[:, 0], c ="blue", s = 5)

plt.imshow(map_image)
plt.savefig(f'./test_with_erosion.png')
plt.show()
plt.clf()

hu = trav_map.wall_heuristic
cmap = sns.cm.rocket_r
ax = sns.heatmap(hu, linewidth=0, cmap=cmap)
plt.scatter(shortest_path_world[:, 1], shortest_path_world[:, 0], c ="blue", s = 5)
plt.savefig(f"./heatmap.png")
plt.show()
# breakpoint()

print(map_image.shape)