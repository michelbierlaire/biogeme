"""
Representation of an ellipse

Michel Bierlaire
Sat Dec 21 16:45:00 2024
"""

from dataclasses import dataclass

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse as MatplotEllipse
import numpy as np


@dataclass
class Ellipse:
    """Contains the information needed to draw an ellipse"""

    center_x: float
    center_y: float
    sin_phi: float
    cos_phi: float
    axis_one: float
    axis_two: float


def draw_ellipse(ellipse: Ellipse, ax=None, **kwargs):
    """
    Draws an ellipse using matplotlib.

    :param ellipse: An Ellipse object containing the parameters of the ellipse.
    :param ax: A matplotlib Axes object. If None, a new figure and axes will be created.
    :param kwargs: Additional keyword arguments passed to matplotlib.patches.Ellipse.
    """
    # Compute the rotation angle of the ellipse in degrees
    phi = np.arctan2(ellipse.sin_phi, ellipse.cos_phi)
    angle = np.degrees(phi)

    # Create the matplotlib ellipse
    patch = MatplotEllipse(
        (ellipse.center_x, ellipse.center_y),  # Center of the ellipse
        width=2 * ellipse.axis_one,  # Major axis (diameter)
        height=2 * ellipse.axis_two,  # Minor axis (diameter)
        angle=angle,  # Rotation angle in degrees
        **kwargs  # Additional styling options
    )

    # Add the ellipse to the axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(patch)

    # Set limits for better visualization
    ax.set_xlim(
        ellipse.center_x - 1.5 * ellipse.axis_one,
        ellipse.center_x + 1.5 * ellipse.axis_one,
    )
    ax.set_ylim(
        ellipse.center_y - 1.5 * ellipse.axis_two,
        ellipse.center_y + 1.5 * ellipse.axis_two,
    )
    ax.set_aspect('equal', adjustable='datalim')

    # Add labels and grid
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.grid(True)

    # Show plot if ax was None
    if ax is None:
        plt.show()
