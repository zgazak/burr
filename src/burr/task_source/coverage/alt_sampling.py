"""Alternative sky sampling implementation that doesn't require healpy."""

import numpy as np
from typing import List, Tuple


def fibonacci_sphere(n_points: int) -> List[Tuple[float, float]]:
    """
    Generate points on a sphere using the Fibonacci spiral method.
    This creates an approximately uniform distribution of points.

    Args:
        n_points: Number of points to generate

    Returns:
        List of (altitude, azimuth) tuples in degrees
    """
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        # Convert to spherical coordinates
        r = np.sqrt(x * x + y * y + z * z)
        theta = np.arccos(y / r)  # Polar angle (from Z)
        phi = np.arctan2(z, x)  # Azimuthal angle

        # Convert to altitude/azimuth
        alt = 90 - np.degrees(theta)  # Convert polar angle to altitude
        az = np.degrees(phi)
        if az < 0:
            az += 360  # Convert to 0-360 range

        points.append((alt, az))

    return points


def angular_separation(alt1: float, az1: float, alt2: float, az2: float) -> float:
    """
    Calculate angular separation between two points in degrees.

    Args:
        alt1, az1: First point's altitude and azimuth in degrees
        alt2, az2: Second point's altitude and azimuth in degrees

    Returns:
        Angular separation in degrees
    """
    # Convert to radians
    alt1, az1 = np.radians([alt1, az1])
    alt2, az2 = np.radians([alt2, az2])

    # Convert alt/az to direction cosines
    x1 = np.cos(alt1) * np.cos(az1)
    y1 = np.cos(alt1) * np.sin(az1)
    z1 = np.sin(alt1)

    x2 = np.cos(alt2) * np.cos(az2)
    y2 = np.cos(alt2) * np.sin(az2)
    z2 = np.sin(alt2)

    # Calculate angular separation
    dot_product = x1 * x2 + y1 * y2 + z1 * z2
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Handle numerical errors
    return np.degrees(np.arccos(dot_product))


def generate_sky_points(
    n_points: int, min_altitude: float = -90, max_altitude: float = 90, min_separation: float = 0
) -> List[Tuple[float, float]]:
    """
    Generate random sky points within altitude limits with minimum separation.

    Args:
        n_points: Desired number of points
        min_altitude: Minimum altitude in degrees
        max_altitude: Maximum altitude in degrees
        min_separation: Minimum angular separation between points in degrees

    Returns:
        List of (altitude, azimuth) tuples in degrees
    """
    points = []
    max_attempts = n_points * 100  # Prevent infinite loops
    attempts = 0

    while len(points) < n_points and attempts < max_attempts:
        # Generate a candidate point
        az = np.random.uniform(0, 360)
        min_alt_rad = np.radians(min_altitude)
        max_alt_rad = np.radians(max_altitude)
        sin_alt = np.random.uniform(np.sin(min_alt_rad), np.sin(max_alt_rad))
        alt = np.degrees(np.arcsin(sin_alt))

        # Check separation from existing points
        if min_separation > 0 and points:
            too_close = False
            for existing_alt, existing_az in points:
                if angular_separation(alt, az, existing_alt, existing_az) < min_separation:
                    too_close = True
                    break
            if too_close:
                attempts += 1
                continue

        points.append((alt, az))

    if len(points) < n_points:
        print(f"Warning: Could only generate {len(points)} points with minimum separation {min_separation}°")

    return points


def plot_sky_points(points: List[Tuple[float, float]], save_path: str = None):
    """
    Plot sky points in a polar projection (top-down view).

    Args:
        points: List of (altitude, azimuth) tuples in degrees
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt

    # Convert alt/az to polar coordinates
    # For polar plot:
    # - radius = 90° - altitude (zenith is center at r=0)
    # - theta = azimuth in radians (0° = North, increasing clockwise)
    alts, azs = zip(*points)
    r = [90 - alt for alt in alts]  # Convert altitude to zenith distance
    theta = np.radians(azs)  # Convert to radians

    # Create polar projection
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="polar")

    # Plot points
    ax.scatter(theta, r, s=40, alpha=0.6)

    # Customize the plot
    ax.set_theta_zero_location("N")  # 0° at North
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_rlim(0, 90)  # Set radial limits (0° to max zenith distance)

    # Set up azimuth ticks every 30 degrees
    az_ticks = np.arange(0, 360, 30)
    ax.set_xticks(np.radians(az_ticks))
    ax.set_xticklabels([f"{int(az)}°" for az in az_ticks])

    # Add altitude circles and labels
    alt_circles = [0, 30, 60]  # Altitudes to mark
    for alt in alt_circles:
        radius = 90 - alt
        # Draw circle at this altitude using polar plot's native circle
        ax.add_artist(
            plt.Circle((0, 0), radius, fill=False, linestyle=":", alpha=0.3, color="gray", transform=ax.transData._b)
        )
        # Add label along the 90° azimuth line (East)
        ax.text(np.radians(90), radius, f"{alt}°", ha="left", va="center", alpha=0.5)

    # Customize grid
    ax.grid(True, alpha=0.3)
    ax.set_rticks([])  # Remove radial ticks

    ax.set_title(f"Sky Coverage Map ({len(points)} points)\nZenith at Center", pad=20)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close()


# Example usage:
if __name__ == "__main__":
    # Generate 1000 points above 20 degrees altitude
    points = generate_sky_points(1000, min_altitude=20)
    plot_sky_points(points)
