import numpy as np


class Frame:
    def __init__(self, u, transform):
        self.u = u
        self.transform = transform
        self.axes = self._compute_axes()
        self.grid = self._compute_grid()

    def _compute_axes(self):
        n = 21
        xt = np.zeros(n)
        xx = np.linspace(-10, 10, n)
        tt = np.linspace(-10, 10, n)
        tx = np.zeros(n)
        xx_, xt_ = self.transform(xx, xt, self.u, inverse=True)
        tx_, tt_ = self.transform(tx, tt, self.u, inverse=True)
        return (xx_, xt_), (tx_, tt_)

    def _compute_grid(self):
        n_points = 201
        x_min, x_max, x_step = -8, 8, 1
        y_min, y_max, y_step = -8, 8, 1
        x_grid = np.arange(x_min, x_max + x_step, x_step)
        y_grid = np.arange(y_min, y_max + y_step, y_step)

        xx_grid, xy_grid = np.repeat(x_grid, n_points), np.tile(
            np.linspace(x_min, x_max, n_points), (x_grid.size, 1)
        )
        yx_grid, yy_grid = np.tile(
            np.linspace(y_min, y_max, n_points), (y_grid.size, 1)
        ), np.repeat(y_grid, n_points)

        xx_grid, xy_grid = self.transform(
            xx_grid.flatten(), xy_grid.flatten(), self.u, inverse=True
        )
        yx_grid, yy_grid = self.transform(
            yx_grid.flatten(), yy_grid.flatten(), self.u, inverse=True
        )

        return (xx_grid, xy_grid), (yx_grid, yy_grid)


def lorentz(x, t, u, c=1, inverse=False):
    u = -u if inverse else u
    beta = u / c
    gamma = 1 / np.sqrt(1 - beta**2)
    x_ = gamma * (x - u * t)
    t_ = gamma * (t - (u * x) / c**2)
    return x_, t_


def galilean(x, t, u, inverse=False):
    u = -u if inverse else u
    x_ = x - u * t
    if isinstance(x, int):
        t_ = t
    else:
        t_ = np.ones(x.size) * t
    return x_, t_


def get_reference_frame():
    n = 21
    xt = np.zeros(n)
    xx = np.linspace(-10, 10, n)
    tt = np.linspace(-10, 10, n)
    tx = np.zeros(n)
    return (xx, xt), (tx, tt)


def get_moving_frame(u, transform):
    (xx, xt), (tx, tt) = get_reference_frame()
    xx_, xt_ = transform(xx, xt, u, inverse=True)
    tx_, tt_ = transform(tx, tt, u, inverse=True)
    return (xx_, xt_), (tx_, tt_)


def create_events(x, t, frame, transform):
    x_, t_ = transform(x, t, frame.u, inverse=frame.u > 0)
    return x_, t_


def get_hyperbolic_time(t, transform):
    velocities = np.linspace(-0.9, 0.9, 20)
    events = np.array([transform(0, t, v, inverse=True) for v in velocities])
    return events[:, 0], events[:, 1]


def get_hyperbolic_space(x, transform):
    velocities = np.linspace(-0.9, 0.9, 20)
    events = np.array([transform(x, 0, v, inverse=True) for v in velocities])
    return events[:, 0], events[:, 1]
