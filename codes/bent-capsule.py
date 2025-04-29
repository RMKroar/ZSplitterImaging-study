import numpy as np
from scipy.interpolate import splprep, splev
from perlin_noise import PerlinNoise
from vedo import Volume, show

def generate_random_spline(length=40, steps=100, seed=0):
    np.random.seed(seed)
    points = np.cumsum(np.random.randn(3, length) * 3, axis=1)
    tck, _ = splprep(points, s=0)
    u = np.linspace(0, 1, steps)
    spline = splev(u, tck)
    return np.array(spline).T  # shape (steps, 3)

def draw_capsule_from_spline(shape, spline_points, radius=5.0):
    volume = np.zeros(shape, dtype=np.float32)
    zz, yy, xx = np.indices(shape)

    for pt in spline_points:
        dist = np.sqrt((xx - pt[0])**2 + (yy - pt[1])**2 + (zz - pt[2])**2)
        sphere_alpha = np.clip((radius - dist) / radius, 0, 1)
        volume = np.maximum(volume, sphere_alpha)

    return volume

def generate_perlin_noise_3d(shape, scale=10, seed=None):
    noise = PerlinNoise(octaves=scale, seed=seed)
    z, y, x = shape
    grid = np.zeros(shape)

    for i in range(z):
        for j in range(y):
            for k in range(x):
                grid[i, j, k] = noise([i/z, j/y, k/x])
    
    # Normalize to [0, 1]
    grid = (grid - grid.min()) / (grid.max() - grid.min())
    return grid

def add_membrane_and_perlin(volume, membrane_thickness=1.5,
                            membrane_alpha=0.8, haze_alpha=0.2,
                            haze_strength=0.2, haze_scale=4, seed=1):
    shape = volume.shape
    perlin = generate_perlin_noise_3d(shape, scale=haze_scale, seed=seed + 123)

    # Compute shell (membrane) zone by erosion
    inner = np.clip(volume - membrane_thickness / radius, 0, 1)
    membrane_zone = volume - inner
    membrane = membrane_zone * membrane_alpha

    # Hazy interior
    haze = inner * (haze_alpha + haze_strength * perlin)

    return np.clip(membrane + haze, 0, 1)

shape = (128, 128, 128)
spline = generate_random_spline(length=50, steps=120, seed=42)
spline = np.clip(spline + np.array(shape) / 2, 0, np.array(shape) - 1)  # Center

radius = 20
volume = draw_capsule_from_spline(shape, spline, radius=radius)
alpha_capsule = add_membrane_and_perlin(volume, membrane_thickness=1.5,
                                        membrane_alpha=0.85, haze_alpha=0.1,
                                        haze_strength=0.25, haze_scale=4, seed=42)

vol = Volume(alpha_capsule)
vol.cmap("bone")
show(vol, axes=1, viewup="z", bg="white")