import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from sklearn.model_selection import train_test_split

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
)
from utils import LABEL_SIZE

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
)
np.random.seed(2021)

TEST_SIZE = .2


# LOAD TERRAIN.
terrain1 = imread("data/SRTM_data_Norway_2.tif")
m, n = terrain1.shape


# SAMPLE TERRAIN.
N = 2000
x = np.random.randint(n, size=(N,))
y = np.random.randint(m, size=(N,))
z = terrain1[y, x]
data = np.column_stack((x, y))


# TRAIN-TEST-SPLIT.
x_train, x_test, y_train, y_test = train_test_split(data, z, test_size=TEST_SIZE)

# PLOT.
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(terrain1, cmap="terrain")
axs[0].set_title("Original.")
axs[1].imshow(terrain1, cmap="terrain")
axs[1].scatter(*x_train.T, s=1, c="k", label="train")
axs[1].scatter(*x_test.T, s=1, c="r", label="test")
axs[1].set_title("Sampled.")
for i in range(len(axs)):
    axs[i].set_xlabel(r"$X$", size=LABEL_SIZE)
    axs[i].set_ylabel(r"$Y$", size=LABEL_SIZE)
plt.tight_layout()
if not os.path.exists("figs/"):
    os.makedirs("figs/")
plt.savefig("figs/terrain.pdf", bbox_inches='tight', format="pdf")
plt.show()
