import matplotlib.pyplot as plt
import numpy as np

from decayshape.lineshapes import CubicInterpolation, LinearInterpolation, QuadraticInterpolation

mass_points = [0.5, 1.0, 1.5, 2.0]
amplitudes = [1.0, 2.0, 1.5, 0.8]
s = np.linspace(0.4, 2.1, 400)

lin = LinearInterpolation(mass_points=mass_points, amplitudes=amplitudes)
quad = QuadraticInterpolation(mass_points=mass_points, amplitudes=amplitudes)
cub = CubicInterpolation(mass_points=mass_points, amplitudes=amplitudes)

y_lin = lin(
    0,
    1,
    s=s,
)
y_quad = quad(0, 1, s=s)
y_cub = cub(0, 1, s=s)

plt.figure(figsize=(8, 5))
plt.plot(s, y_lin, label="Linear")
plt.plot(s, y_quad, label="Quadratic")
plt.plot(s, y_cub, label="Cubic")
plt.scatter(mass_points, amplitudes, c="k", zorder=3, label="Points")
plt.xlabel("s")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("plot_interpolations.png", dpi=150)
plt.show()
