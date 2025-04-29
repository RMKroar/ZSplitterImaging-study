import numpy as np
from vedo import Volume, show
from perlin_noise import PerlinNoise

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

def create_dented_balloon(shape=(96, 96, 96), base_radius=35, dent_amplitude=4.0, dent_scale=4,
                          membrane_thickness=2.5, membrane_alpha=0.8,
                          interior_alpha=0.2, haze_strength=0.2, haze_scale=4.0, seed=0):
    center = np.array(shape) / 2
    z, y, x = np.indices(shape)

    # Distance grid
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    # Perlin noise for surface deformation (dents)
    dent_noise = generate_perlin_noise_3d(shape, scale=dent_scale, seed=seed)
    local_radius = base_radius + (dent_amplitude * (dent_noise - 0.5))  # centered at base_radius

    # Membrane
    membrane_mask = np.logical_and(
        distances >= (local_radius - membrane_thickness),
        distances <= local_radius
    )
    membrane = membrane_mask.astype(float) * membrane_alpha

    # Interior: haze via Perlin noise
    haze_noise = generate_perlin_noise_3d(shape, scale=haze_scale, seed=seed + 100)
    interior_mask = distances < (local_radius - membrane_thickness)
    haze = interior_alpha + haze_strength * haze_noise
    interior = interior_mask.astype(float) * haze

    # Combine
    alpha = membrane + interior
    alpha = np.clip(alpha, 0, 1)

    return alpha

alpha_dented_balloon = create_dented_balloon(
    shape=(96, 96, 96),
    base_radius=35,
    dent_amplitude=5.0,
    dent_scale=3,
    membrane_thickness=2.5,
    membrane_alpha=0.9,
    interior_alpha=0.15,
    haze_strength=0.25,
    haze_scale=6,
    seed=42
)

print(alpha_dented_balloon.shape)

# vedo expects 3D volumes where values represent intensity or opacity
# Let's directly use the alpha_sphere

# Create a Volume
vol = Volume(alpha_dented_balloon)
vol.cmap("bone")
show(vol, axes=1, viewup="z", bg="white")