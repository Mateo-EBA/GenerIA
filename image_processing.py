#%% --- --- --- --- --- --- --- --- ---
# Imports
import random
import numpy as np

import cv2

#%% --- --- --- --- --- --- --- --- ---
# Superposed patches
def get_superposed_patches_coords(img_width:int, img_height:int, count_x:int, count_y:int):
    """
    Get the top-left and bot-right corner coordinates for count_x by count_y
    half superposing crops of a given image size.
    
    Args:
        img_width (int): the width of the image.
        img_height (int): the height of the image.
        count_x (int): an integer representing the number of smaller images to create in the X dimension.
        count_y (int): an integer representing the number of smaller images to create in the Y dimension.
    
    Returns:
        A list of tuples, where each tuple contains:
            An integer representing the x-coordinate of the top-left corner of the cropped image.
            
            An integer representing the y-coordinate of the top-left corner of the cropped image.
            
            An integer representing the x-coordinate of the bot-right corner of the cropped image.
            
            An integer representing the y-coordinate of the bot-right corner of the cropped image.
    """
    
    crop_height = img_height / (count_y + 1)
    crop_width = img_width / (count_x + 1)

    crop_coords_list = []
    for h_idx in range(count_y):
        for w_idx in range(count_x):
            if h_idx == (count_y - 1) and w_idx == (count_x - 1):
                start_y = h_idx * crop_height
                start_x = w_idx * crop_width
                end_x = img_width
                end_y = img_height

            elif h_idx != (count_y - 1) and w_idx == (count_x - 1):
                start_y = h_idx * crop_height
                start_x = w_idx * crop_width
                end_x = img_width
                end_y = (h_idx + 2) * crop_height     

            elif h_idx == (count_y - 1) and w_idx != (count_x - 1):
                start_y = h_idx * crop_height
                start_x = w_idx * crop_width
                end_x = (w_idx + 2) * crop_width
                end_y = img_height

            else:
                start_y = h_idx * crop_height
                start_x = w_idx * crop_width
                end_x = (w_idx + 2) * crop_width
                end_y = (h_idx + 2) * crop_height   
            
            crop_coords_list.append((start_x, start_y, end_x, end_y)) 
    
    return crop_coords_list

def get_superposed_patches(img, count_x:int, count_y:int):
    """
    Extracts superposed patches from an input image.

    Args:
        img (numpy.ndarray or array like with shape attribute): The input image. Expected in CHW
        count_x (int): The number of patches along the x-axis. Default is 3.
        count_y (int): The number of patches along the y-axis. Default is 3.

    Returns:
        list: A list of tuples, each containing a patch image and its corresponding
        coordinates on the original image (top-left x, top-left y, bottom-right x, bottom-right y).
    """
    img_height, img_width = img.shape[:2]
    
    crops_coords = get_superposed_patches_coords(img_width, img_height, count_x, count_y)

    crop_image_list = []
    for crop_coords in crops_coords:
        start_x = round(crop_coords[0])
        start_y = round(crop_coords[1])
        end_x = round(crop_coords[2])
        end_y = round(crop_coords[3])
        
        crop_img = img[start_y:end_y, start_x:end_x, :].copy()
        crop_image_list.append((crop_img, start_x, start_y, end_x, end_y))

    return crop_image_list

#%% --- --- --- --- --- --- --- --- ---
# Filters
def percentile_truncation_normalization(img, pmin=5, pmax=99):
    """
    Clip the image between the given percentiles and normalize between 0 and 1.

    Args:
        img (np.ndarray): The input image.
        pmin (int, optional): The percentile value at the lower end of the truncation range. Defaults to 5.
        pmax (int, optional): The percentile value at the upper end of the truncation range. Defaults to 99.

    Returns:
        np.ndarray: The normalized image.
    """
    Pmin = np.percentile(img, pmin)
    Pmax = np.percentile(img, pmax)
    truncated = np.clip(img, Pmin, Pmax)  
    normalized = (truncated - Pmin)/(Pmax - Pmin)
    return normalized

def img_clahe_filter(img, clip_limit:int, tile_grid_size:tuple[int,int]):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) filter to an image.

    Args:
        img (np.ndarray): The input image.
        clip_limit (int): The clip limit parameter for the CLAHE filter.
        tile_grid_size (tuple): The tile grid size parameter for the CLAHE filter.

    Returns:
        np.ndarray: The filtered image.

    Warning:
        If you only need to apply the filter to a single image, you can use this function directly.

        However, the `cv2.createCLAHE` function is expensive in terms of memory and computation.
        It is recommended to create the filter object once and reuse it for multiple images,
        rather than creating it every time this function is called.

        For example, if you are applying the filter to multiple frames of a video,
        it would be more efficient to create the filter object once and apply it to each frame,
        like this:

        ```python
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            for frame in frames:
                clahe_frame = clahe.apply(frame)
        ```
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(img)
    return cl

#%% --- --- --- --- --- --- --- --- ---
# Image checks
def check_2D_vicinity_above_threshold(arr, threshold:float, min_amount_above_in_vicinity:int, vicinity_size:int|tuple[int,int]=3):
    """
    Check if a 2D array has a vicinity where the number of elements above a threshold exceeds a specified amount.

    Args:
        arr (np.ndarray): The 2D array to check.
        threshold (float): The threshold value to compare elements against.
        min_amount_above_in_vicinity (int): The minimum number of elements above the threshold required in the vicinity.
        vicinity_size (int, tuple[int,int], optional): The size of the vicinity window. Square if single value given. Defaults to 3.

    Returns:
        bool: True if a vicinity with the specified conditions is found, False otherwise.
    """
    min_amount_above_in_vicinity = min_amount_above_in_vicinity - 1
    if isinstance(vicinity_size, int):
        vicinity_x = vicinity_size
        vicinity_y = vicinity_size
    else:
        vicinity_x = vicinity_size[0]
        vicinity_y = vicinity_size[1]
    offset_x = vicinity_x-1
    offset_y = vicinity_y-1
    rows, cols = arr.shape
    for i in range(rows - offset_x):
        for j in range(cols - offset_y):
            vicinity_values = arr[i:i+vicinity_x, j:j+vicinity_y]
            if np.sum(vicinity_values > threshold) > min_amount_above_in_vicinity:
                return True
    return False

#%% --- --- --- --- --- --- --- --- ---
# Image creation
def generate_random_binary_np_image(W:int, H:int, C:int, amount_ones:int=-1):
    """
    This function generates a random binary image with a given width, height, and number of channels.

    Args:
        W (int): The width of the image.
        H (int): The height of the image.
        C (int): The number of channels in the image.
        amount_ones (int, optional): The number of 1's to include in the image. If not provided, a random number of 1's will be generated.

    Returns:
        np.ndarray: A numpy array representing the binary image.
    """
    amount_elements = H * W * C
    if amount_ones < 0:
        amount_ones = np.random.randint(0, amount_elements-1)
    matrix = np.zeros((amount_elements, 1), dtype=int)
    random_indexes = random.sample(range(amount_elements), amount_ones)
    for i in random_indexes:
        matrix[i] = 1
    return matrix.reshape((H, W, C))
