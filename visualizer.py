"""
Plotting utility
"""
import math
import numpy as np
import torch
import matplotlib.pyplot as plt


def _build_display_grid(features: torch.Tensor | np.ndarray,
                        rows: int = 0) -> np.ndarray:
    """
    Build and return the display grid (as a 2D NumPy array) from a set of features.

    Parameters
    ----------
    features : np.ndarray
        A 2D array of shape (L, M) where each column is one feature (flattened square).
    rows : int, optional
        Number of rows in the displayed figure. If <= 0, a square layout is chosen automatically.

    Returns
    -------
    display_grid : np.ndarray
        2D array representing the tiled features (values in [-1, 1]).
    """

    # Convert torch.Tensor to np.ndarray if needed
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    elif not isinstance(features, np.ndarray):
        raise TypeError("display_network can only take a torch.Tensor or a numpy.ndarray.")

    # Determine patch size and validate
    L, M = features.shape  # M features, each of size L
    psize = int(math.sqrt(L))
    if psize * psize != L:
        raise ValueError("Each feature column must represent a square patch.")
    
    # Decide the grid layout
    if rows <= 0:
        rows = int(math.ceil(math.sqrt(M)))
    cols = int(math.ceil(M / rows))

    # Normalize each feature column so that its max absolute value is 1
    abs_max = np.max(np.abs(features), axis=0)
    abs_max[abs_max == 0] = 1.0  # Avoid division by zero
    features_scaled = features / abs_max
    
    # Reshape into (psize, psize, M) for easier indexing
    features_scaled = features_scaled.reshape(psize, psize, M)

    # Construct the display grid
    border = 1
    grid_height = border + rows * (psize + border)
    grid_width = border + cols * (psize + border)
    display_grid = -np.ones((grid_height, grid_width))

    # Fill the grid
    for i in range(M):
        r = i // cols
        c = i % cols
        top = border + r * (psize + border)
        left = border + c * (psize + border)
        display_grid[top:top+psize, left:left+psize] = features_scaled[:, :, i]
    return display_grid


def display_network(features: torch.Tensor | np.ndarray,
                    rows: int = 0,
                    figsize=(10, 10)) -> None:
    """
    Display the learned features in a grid.

    Each column of 'features' is assumed to be one feature of size psize * psize.
    By default, the grid of subplots will be approximately square (rows ≈ sqrt(M)).
    
    Parameters
    ----------
    features : torch.Tensor or np.ndarray
        A 2D array of shape (L, M), where each column is a flattened psize x psize patch.
    rows : int, optional
        Number of rows in the displayed figure. If <= 0, a square layout is chosen automatically.
    figsize : tuple, optional
        Size of the matplotlib figure.
    """

    display_grid = _build_display_grid(features, rows)
    plt.figure(figsize=figsize)
    plt.imshow(display_grid, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
    plt.show()


def get_network_as_image(features: torch.Tensor | np.ndarray,
                         rows: int = 0,
                         figsize=(10, 10)) -> np.ndarray:
    """
    Return the learned features plot as an RGB image (NumPy array), which can then be saved.

    Parameters
    ----------
    features : torch.Tensor or np.ndarray
        A 2D array of shape (L, M) where each column is a flattened psize x psize patch.
    rows : int, optional
        Number of rows in the displayed figure. If <= 0, a square layout is chosen automatically.
    figsize : tuple, optional
        Size of the matplotlib figure.

    Returns
    -------
    image : np.ndarray
        An H x W x 3 RGB image that represents the tiled features.
    """
    # Create the image to plot
    display_grid = _build_display_grid(features, rows)

    # Create a figure (no interactive display)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(display_grid, cmap='gray', vmin=-1, vmax=1)
    ax.axis('off')

    # Render the figure to an RGBA buffer
    fig.canvas.draw()  # draw the canvas, cache the renderer
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = buf.reshape((h, w, 4))
    image = np.transpose(image, (2, 0, 1))

    # Close the figure to free resources
    plt.close(fig)

    return image
