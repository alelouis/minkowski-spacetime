import matplotlib.pyplot as plt
import os

from minkowski import *

plt.style.use(os.path.abspath("./mplrc"))

c = 1.0
u = 0.5
beta = u / c
gamma = 1 / np.sqrt(1 - beta**2)

ref_frame = Frame(u=0, transform=lorentz)
moving_frame = Frame(u=u, transform=lorentz)

(xx, xt), (tx, tt) = ref_frame.axes
(xx_, xt_), (tx_, tt_) = moving_frame.axes

(xx_grid, xy_grid), (yx_grid, yy_grid) = ref_frame.grid
(xx_grid_, xy_grid_), (yx_grid_, yy_grid_) = moving_frame.grid

ax, at = create_events(5, 0, ref_frame, lorentz)

htx, htt = get_hyperbolic_time(5, lorentz)
hxx, hxt = get_hyperbolic_space(5, lorentz)

plt.figure(dpi=120, figsize=(6, 6))

plt.plot(
    xx,
    xt,
    c="yellow",
    marker="o",
    markersize=4,
    linewidth=1,
    markerfacecolor="none",
    markeredgewidth=0.5,
)
plt.plot(
    tx,
    tt,
    c="yellow",
    marker="o",
    markersize=4,
    linewidth=1,
    markerfacecolor="none",
    markeredgewidth=0.5,
)

plt.plot(
    xx_,
    xt_,
    c="cyan",
    marker="o",
    markersize=4,
    linewidth=1,
    markerfacecolor="none",
    markeredgewidth=0.5,
)
plt.plot(
    tx_,
    tt_,
    c="cyan",
    marker="o",
    markersize=4,
    linewidth=1,
    markerfacecolor="none",
    markeredgewidth=0.5,
)

plt.scatter(ax, at, c="r")

plt.scatter(xx_grid, xy_grid, c="yellow", edgecolor="none", s=0.2, alpha=1)
plt.scatter(yx_grid, yy_grid, c="yellow", edgecolor="none", s=0.2, alpha=1)
plt.scatter(xx_grid_, xy_grid_, c="cyan", edgecolor="none", s=0.2, alpha=1)
plt.scatter(yx_grid_, yy_grid_, c="cyan", edgecolor="none", s=0.2, alpha=1)

plt.plot(htx, htt, c="lightcoral")
plt.plot(hxx, hxt, c="lightcoral")

plt.xlim([-8, 8])
plt.ylim([-8, 8])

plt.savefig("example.png", dpi=300)
plt.show()
