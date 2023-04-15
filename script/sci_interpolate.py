import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev

import json

with open('../data/example_14th_59.json', 'r', encoding='utf-8') as f:
    val = json.load(f)

print(len(val))

n0 = np.array(val)
n1 = n0[::30, :]
n2 = np.r_[n1, n1[0, :].reshape(1, 2)]
sz = n2.shape[0]

csX = CubicSpline(np.arange(sz), n2[:, 0], bc_type='periodic')
csY = CubicSpline(np.arange(sz), n2[:, 1], bc_type='periodic')

space = np.linspace(0, sz, 5000)
fig, ax = plt.subplots()
# ax.scatter(n2[:, 0], n2[:, 1])
x_plot, y_plot = csX(space), csY(space)
ax.plot(x_plot, y_plot, linewidth=1)
print(x_plot.shape, y_plot.shape)
# for i in range(100):
#     ax.annotate(f"{i}", (x_plot[i], y_plot[i]))
plt.xlabel((80.0, 280.0))
plt.ylabel((150.0, 350.0))
plt.show()

# if __name__ == '__main__':
# with open('../data/examples10_100x100.json', 'r', encoding='utf-8') as f:
#     val: list = json.load(f)[4]
#     n0 = np.array(val)
# x, y = np.r_[n0[:, 0], n0[0, 0]], np.r_[n0[:, 1], n0[0, 1]]
# tck, u = splprep([x, y], s=0, per=1)
# xi, yi = splev(np.linspace(-50, 50, 100), tck)
# fig, ax = plt.subplots(1, 1)
# ax.plot(x, y)
# plt.show()
# cs = CubicSpline(x, y)
# xs = np.arange(-50.0, 50.0, 1000)
# fix, ax = plt.subplots()
# ax.plot(x, y, linewidth='0.5')
# plt.show()
