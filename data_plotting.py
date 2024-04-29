#%% --- --- --- --- --- --- --- --- ---
# Imports
import matplotlib.pyplot as plt

#%% --- --- --- --- --- --- --- --- ---
# Functions
def plot_image_borderless(x, cmap="gray"):
    """
    Plot an image without borders.

    Args:
        x (array-like): The image data to plot.
        cmap (str, optional): The colormap to use for the image. Defaults to "gray".
    """
    plt.figure()
    plt.axis("off")
    plt.imshow(x, cmap=cmap)

def plot_image_borderless_with_title(x, title:str, cmap="gray"):
    """
    Plot an image with a label without borders.

    Args:
        x (array-like): The image data to plot.
        title (str): The label for the image.
        cmap (str, optional): The colormap to use for the image. Defaults to "gray".
    """
    plt.figure()
    plt.axis("off")
    plt.imshow(x, cmap=cmap)
    plt.title(title)

    
