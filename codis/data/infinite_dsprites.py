import cmath
from math import atan2, pi
from random import random
from matplotlib import pyplot as plt
import numpy as np
import bezier
from scipy.interpolate import splprep, splev


def convex_hull(pts):
    xleftmost, yleftmost = min(pts)
    by_theta = [(atan2(x - xleftmost, y - yleftmost), x, y) for x, y in pts]
    by_theta.sort()
    as_complex = [complex(x, y) for _, x, y in by_theta]
    hull = as_complex[:2]
    for pt in as_complex[2:]:
        # Perp product.
        while ((pt - hull[-1]).conjugate() * (hull[-1] - hull[-2])).imag < 0:
            hull.pop()
        hull.append(pt)
    return [(pt.real, pt.imag) for pt in hull]


def interpolate_spline(points):
    points = np.array(points).T
    print(points.shape)
    tck, u = splprep(points, u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x, y = splev(u_new, tck, der=0)
    return np.array([x, y])


def interpolate_bezier(points, num_points=100):
    points = np.array(points).T
    curve = bezier.Curve(nodes=points, degree=points.shape[1] - 1)
    points = curve.evaluate_multi(np.linspace(0, 1, num_points))
    return points


def generate_shape(n=6):
    points = [(random() + 0.8) * cmath.exp(2j * pi * i / 7) for i in range(n)]
    points = convex_hull([(p.real, p.imag) for p in points])
    points = interpolate_spline(points)
    return points


def main():
    points = generate_shape()
    plt.plot(points[0], points[1], label="spline", color="red")
    plt.show()


if __name__ == "__main__":
    main()
