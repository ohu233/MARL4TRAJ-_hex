import math
import pickle
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.tools import mapdata_to_modelmatrix
from utils.geo_utils import grid_to_mercator, full_grid_bounds_mercator, hex_to_mercator, hex_bounds_mercator
from utils.basemap import set_full_extent
from utils.hex_utils import load_hex_mapdata, load_hex_mapdata_raw, code_to_mode_matrices

Image.MAX_IMAGE_PIXELS = None

# 1) load data
with open(r"data\GridModesAdjacentRealworld.pkl", "rb") as f:
    mapdata = pickle.load(f)

matrice = mapdata_to_modelmatrix(mapdata, 529, 564)

# 2) base config
modes = ["TG", "GG", "GSD", "TS"]
mode_colors = {
    "TG": "orange",
    "GG": "blue",
    "GSD": "green",
    "TS": "red",
}

# 3) precompute points for each mode
mode_points = {}
for mode in modes:
    matrix = matrice[mode]
    x, y = zip(*[(i, j) for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j] == 1])
    mode_points[mode] = (x, y)

h = len(matrice[modes[0]])
w = len(matrice[modes[0]][0])

# 4) generate all non-empty combinations
all_combos = []
for r in range(1, len(modes) + 1):
    all_combos.extend(combinations(modes, r))

# 5) save one figure per combination (keep original style)
for combo in all_combos:
    fig, ax = plt.subplots(figsize=(20, 20))

    set_full_extent(ax)

    # overlay all modes in this combination
    for mode in combo:
        grid_xs, grid_ys = mode_points[mode]
        merc_x, merc_y = grid_to_mercator(np.array(grid_xs), np.array(grid_ys))
        ax.scatter(merc_x, merc_y, s=1 / 100, alpha=1, c=mode_colors[mode], marker='o')

    ax.axis('off')

    combo_name = "+".join(combo)
    plt.tight_layout()
    plt.savefig(f'figure/{combo_name}.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

print(f"Saved {len(all_combos)} combination figures.")