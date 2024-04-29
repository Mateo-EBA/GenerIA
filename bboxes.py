#%% --- --- --- --- --- --- --- --- ---
# Imports
import numpy as np
from abc import ABC, abstractmethod

#%% --- --- --- --- --- --- --- --- ---
# Simple Computations
def get_box_area(box) -> float:
    """
    Calculate the area of a box given its coordinates.

    Args:
        box (list): A list containing the x and y coordinates of the top-left and bottom-right corners of the box.

    Returns:
        float: The area of the box calculated as (width * height).
    """
    return (box[2] - box[0]) * (box[3] - box[1])

def get_boxes_iou(box_a, box_b) -> float:
    """
    Calculate the Intersection over Union (IoU) of two boxes.

    Args:
        box_a (list): A list containing the x and y coordinates of the top-left and bottom-right corners of box A.
        box_b (list): A list containing the x and y coordinates of the top-left and bottom-right corners of box B.

    Returns:
        float: The Intersection over Union (IoU) value of the two boxes.
    """
    # determine the intersection box
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # compute the area of the intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both original boxes
    boxAArea = get_box_area(box_a)
    boxBArea = get_box_area(box_b)

    # compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return IoU
    return iou

def get_boxes_overlaps(box_a, box_b):
    """
    Calculates the percentage of one box contained in another box.

    Args:
        box_a (list): A tensor of 4 float values representing the xmin, ymin, xmax, ymax coordinates of box A.
        box_b (list): A tensor of 4 float values representing the xmin, ymin, xmax, ymax coordinates of box B.

    Returns:
        tuple[float, float]: The percentage of box A contained in box B and the percentage of box B contained in box A.
    """
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    # Calculate the area of the intersection rectangle
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    # Calculate the area of box_a
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    # Calculate the area of box_b
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    # Calculate the percentage of box_a contained in box_b
    box_a_overlap = intersection_area / box_a_area
    # Calculate the percentage of box_b contained in box_a
    box_b_overlap = intersection_area / box_b_area
    return box_a_overlap, box_b_overlap

def get_boxes_min_distance(box_a, box_b):
    """
    Calculate the minimum distance between two boxes.

    Args:
        box_a (list): A list containing the x and y coordinates of the top-left and bottom-right corners of box A.
        box_b (list): A list containing the x and y coordinates of the top-left and bottom-right corners of box B.

    Returns:
        float: The minimum distance between the two boxes.
    """
    # Check for overlap
    box_a_overlap, box_b_overlap = get_boxes_overlaps(box_a, box_b)
    if box_a_overlap > 0 or box_b_overlap > 0:
        return 0
    
    # Extract coordinates
    min_x_a, min_x_b = box_a[0], box_b[0]
    min_y_a, min_y_b = box_a[1], box_b[1]
    max_x_a, max_x_b = box_a[2], box_b[2]
    max_y_a, max_y_b = box_a[3], box_b[3]
    
    # Helper functions to calculate distances
    def get_min_x_dist(bx_a, bx_b):
        return min(
            abs(bx_a[0] - bx_b[0]),
            abs(bx_a[0] - bx_b[2]),
            abs(bx_a[2] - bx_b[0]),
            abs(bx_a[2] - bx_b[2]),
        )
    def get_min_y_dist(bx_a, bx_b):
        return min(
            abs(bx_a[1] - bx_b[1]),
            abs(bx_a[1] - bx_b[3]),
            abs(bx_a[3] - bx_b[1]),
            abs(bx_a[3] - bx_b[3]),
        )
    
    # Calculate minimum distance based on overlaps
    # X overlap
    if (min_x_a < min_x_b and min_x_b < max_x_a) or (min_x_a < max_x_b and max_x_b < max_x_a):
        dist = get_min_y_dist(box_a, box_b)
    # Y overlap
    elif (min_y_a < min_y_b and min_y_b < max_y_a) or (min_y_a < max_y_b and max_y_b < max_y_a):
        dist = get_min_x_dist(box_a, box_b)
    # No overlap
    else:
        dist = min(
            max(
                abs(box_a[0] - box_b[2]),
                abs(box_a[1] - box_b[3]),
            ),
            max(
                abs(box_a[2] - box_b[0]),
                abs(box_a[3] - box_b[1]),
            ),
        )
    return dist

#%% --- --- --- --- --- --- --- --- ---
# Box Merges
class BoxMergeStrategy(ABC):
    """
    Abstract base class for box merge strategies.

    Methods:
        merge_boxes(self, box_a: List[float], box_b: List[float]) -> List[float]:
            Merges two boxes using the specific strategy.
    """
    @abstractmethod
    def merge_boxes(self, box_a, box_b, *args, **kwargs):
        """
        Merges two boxes using the specific strategy.

        Args:
            box_a (List[float]): The first box to merge.
            box_b (List[float]): The second box to merge.

        Returns:
            List[float]: The merged box.
        """
        pass

def merge_boxes_by_max_coords(box_a, box_b):
    """
    Merge two boxes by expanding box A to include the maximum coordinates of both boxes.

    Args:
        box_a (list): A list containing the x and y coordinates of the top-left and bottom-right corners of box A.
        box_b (list): A list containing the x and y coordinates of the top-left and bottom-right corners of box B.

    Returns:
        list: The merged box with expanded coordinates to encompass both input boxes.
    """
    box_a[0] = min(box_a[0], box_b[0])
    box_a[1] = min(box_a[1], box_b[1])
    box_a[2] = max(box_a[2], box_b[2])
    box_a[3] = max(box_a[3], box_b[3])
    return box_a

class MaxCoord_BoxMergeStrategy(BoxMergeStrategy):
    """
    Merges two boxes by taking the maximum coordinates.

    Methods:
        merge_boxes(self, box_a: List[float], box_b: List[float]) -> List[float]:
            Merges two boxes by taking the maximum coordinates.
    """
    def merge_boxes(self, box_a, box_b, *args, **kwargs):
        """
        Merges two boxes by taking the maximum coordinates.

        Args:
            box_a (List[float]): The first box to merge.
            box_b (List[float]): The second box to merge.

        Returns:
            List[float]: The merged box.
        """
        return merge_boxes_by_max_coords(box_a, box_b)

def merge_boxes_by_weighted_mean(box_a, box_b, weight_a:float, weight_b:float):
    """
    Merge two boxes by calculating the weighted mean of their coordinates.

    Args:
        box_a (list): A list containing the x and y coordinates of the top-left and bottom-right corners of box A.
        box_b (list): A list containing the x and y coordinates of the top-left and bottom-right corners of box B.
        weight_a (float): The weight for box A.
        weight_b (float): The weight for box B.

    Returns:
        list: The merged box with coordinates calculated as the weighted mean of both input boxes.
    """
    total_weight = weight_a + weight_b
    box_a[0] = ((box_a[0] * weight_a) + (box_b[0] * weight_b)) / total_weight
    box_a[1] = ((box_a[1] * weight_a) + (box_b[1] * weight_b)) / total_weight
    box_a[2] = ((box_a[2] * weight_a) + (box_b[2] * weight_b)) / total_weight
    box_a[3] = ((box_a[3] * weight_a) + (box_b[3] * weight_b)) / total_weight
    return box_a

class ScoreWeightedMean_BoxMergeStrategy(BoxMergeStrategy):
    """
    Merges boxes by taking a weighted mean of their coordinates, where the weights are the scores of the boxes.

    Methods:
        merge_boxes(self, box_a: List[float], box_b: List[float], weight_a: float, weight_b: float) -> List[float]:
            Merges two boxes by taking a weighted mean of their coordinates, where the weights are the scores of the boxes.
    """
    def merge_boxes(self, box_a, box_b, weight_a:float, weight_b:float, *args, **kwargs):
        """
        Merges two boxes by taking a weighted mean of their coordinates, where the weights are the scores of the boxes.

        Args:
            box_a (List[float]): The first box to merge.
            box_b (List[float]): The second box to merge.
            weight_a (float): The weight of the first box.
            weight_b (float): The weight of the second box.

        Returns:
            List[float]: The merged box.
        """
        return merge_boxes_by_weighted_mean(box_a, box_b, weight_a, weight_b)

def merge_boxes_by_area_weighted_mean(box_a, box_b):
    """
    Merge two boxes by calculating the area-weighted mean of their coordinates.

    Args:
        box_a (list): A list containing the x and y coordinates of the top-left and bottom-right corners of box A.
        box_b (list): A list containing the x and y coordinates of the top-left and bottom-right corners of box B.

    Returns:
        list: The merged box with coordinates calculated as the area-weighted mean of both input boxes.
    """
    weight_a = get_box_area(box_a)
    weight_b = get_box_area(box_b)
    box_a = merge_boxes_by_weighted_mean(box_a, box_b, weight_a, weight_b)
    return box_a

class AreaWeightedMean_BoxMergeStrategy(BoxMergeStrategy):
    """
    Merges boxes by taking a weighted mean of their coordinates, where the weights are the areas of the boxes.

    Methods:
        merge_boxes(self, box_a: List[float], box_b: List[float]) -> List[float]:
            Merges two boxes by taking a weighted mean of their coordinates, where the weights are the areas of the boxes.
    """
    def merge_boxes(self, box_a, box_b, *args, **kwargs):
        """
        Merges two boxes by taking a weighted mean of their coordinates, where the weights are the areas of the boxes.

        Args:
            box_a (List[float]): The first box to merge.
            box_b (List[float]): The second box to merge.

        Returns:
            List[float]: The merged box.
        """
        return merge_boxes_by_area_weighted_mean(box_a, box_b)
    
def merge_boxes_by_weight_and_area_weighted_mean(box_a, box_b, weight_a:float, weight_b:float):
    """
    Merge two boxes by calculating the weight and area-weighted mean of their coordinates.

    Args:
        box_a (list): A list containing the x and y coordinates of the top-left and bottom-right corners of box A.
        box_b (list): A list containing the x and y coordinates of the top-left and bottom-right corners of box B.
        weight_a (float): The weight for box A.
        weight_b (float): The weight for box B.

    Returns:
        list: The merged box with coordinates calculated as the weight and area-weighted mean of both input boxes.
    """
    weight_a = weight_a * get_box_area(box_a)
    weight_b = weight_b * get_box_area(box_b)
    box_a = merge_boxes_by_weighted_mean(box_a, box_b, weight_a, weight_b)
    return box_a

class ScoreAreaWeightedMean_BoxMergeStrategy(BoxMergeStrategy):
    """
    Merges boxes by taking a weighted mean of their coordinates, where the weights are the areas of the boxes and the scores of the boxes.

    Methods:
        merge_boxes(self, box_a: List[float], box_b: List[float], weight_a: float, weight_b: float) -> List[float]:
            Merges two boxes by taking a weighted mean of their coordinates, where the weights are the areas of the boxes and the scores of the boxes.
    """
    def merge_boxes(self, box_a, box_b, weight_a:float, weight_b:float, *args, **kwargs):
        """
        Merges two boxes by taking a weighted mean of their coordinates, where the weights are the areas of the boxes and the scores of the boxes.

        Args:
            box_a (List[float]): The first box to merge.
            box_b (List[float]): The second box to merge.
            weight_a (float): The weight of the first box.
            weight_b (float): The weight of the second box.

        Returns:
            List[float]: The merged box.
        """
        return merge_boxes_by_weight_and_area_weighted_mean(box_a, box_b, weight_a, weight_b)
    
#%% --- --- --- --- --- --- --- --- ---
# Non-Max-Supression
def nms(boxes,
        scores,
        classes=None,
        iou_thresh:float=0.2,
        box_merge_strategy=MaxCoord_BoxMergeStrategy(),
        contained_percentage_thresh:float=0.5,
        max_recursive_calls:int=3):
    """
    Apply Non-Maximum Suppression (NMS) to a list of boxes based on their scores and classes.

    Args:
        boxes (list): A list of lists containing the x and y coordinates of the top-left and bottom-right corners of each box.
        scores (list): A list of scores corresponding to each box.
        classes (list, optional): A list of classes corresponding to each box. Defaults to None.
        iou_thresh (float, optional): The Intersection over Union (IoU) threshold for merging boxes. Defaults to 0.2.
        box_merge_strategy (BoxMergeStrategy, optional): The strategy for merging boxes. Defaults to MaxCoordBoxMergeStrategy().
        contained_percentage_thresh (float, optional): The threshold for the contained percentage of boxes for merging. Defaults to 0.5.
        max_recursive_calls (int, optional): The maximum number of recursive calls for the NMS algorithm. Defaults to 3.

    Returns:
        tuple: A tuple containing the final boxes, scores, classes, and indices of the boxes.
    """
    def check_merge_condition(box_a, box_b):
        iou = get_boxes_iou(box_a, box_b)
        if iou > iou_thresh:
            return True
        bcp_a, bcp_b = get_boxes_overlaps(box_a, box_b)
        return (bcp_a > contained_percentage_thresh) or (bcp_b > contained_percentage_thresh)
    
    box_merge_process = box_merge_strategy.merge_boxes
    
    def _apply_maximizing_nms(boxes, scores, classes=None):
        # Initialize
        amount = len(scores)
        sorted_scores, indices = np.sort(scores)[::-1]
        sorted_boxes = boxes[indices]
        if classes is not None: sorted_classes = classes[indices]
        removed_indexes = [False]*amount
        final_boxes, final_scores, final_classes, max_score_idxs = [], [], [], []
        
        # For each box
        for i in range(amount):
            # If its not removed already
            if removed_indexes[i]: continue
            
            # If it isn't the last one
            if i == amount-1:
                final_boxes.append(sorted_boxes[i])
                final_scores.append(sorted_scores[i])
                if classes is not None: final_classes.append(sorted_classes[i])
                max_score_idxs.append(i)
                break
            
            # Get the correspoding coordinates and score
            current_box = sorted_boxes[i]
            current_score = sorted_scores[i]
            
            # For each next box (less score)
            for j in range(i+1,amount):
                
                # If classes are given, skip if they are different
                if classes is not None and sorted_classes[i] != sorted_classes[j]: continue
                
                # Get the correspoding coordinates and score
                box_j = sorted_boxes[j]
                score_j = sorted_scores[j]
                
                # Check if they should be merged
                merge = check_merge_condition(current_box, box_j)
                
                # If they should, add the index to the removed and perform merge
                if merge:
                    removed_indexes[j] = True
                    current_box = box_merge_process(current_box, box_j, current_score, score_j)
            
            # Add to the final results
            final_boxes.append(current_box)
            final_scores.append(current_score)
            if classes is not None: final_classes.append(sorted_classes[i])
            max_score_idxs.append(i)

        # Done
        return final_boxes, final_scores, final_classes, max_score_idxs
    
    previous_count = len(scores)
    fbx, fsc, fcl, idx = boxes, scores, classes, list(range(previous_count))
    iteration = 0
    while iteration < max_recursive_calls:
        fbx, fsc, fcl, idx = _apply_maximizing_nms(fbx, fsc, fcl)
        current_count = len(fsc)
        if current_count == previous_count: break
        previous_count = current_count
    return fbx, fsc, fcl, idx

#%% --- --- --- --- --- --- --- --- ---
# Merge by distance
def apply_merge_by_distance(boxes, distance_thresh:int=30, max_recursive_calls:int=3):
    """
    Applies merge by distance algorithm to a list of boxes.

    Merges boxes that are closer than a specified threshold.

    Args:
        boxes (List[List[float]]): A list of boxes, where each box is represented as a list of four floats [x1, y1, x2, y2].
        distance_thresh (int, optional): The distance threshold for merging boxes. Defaults to 30.
        max_recursive_calls (int, optional): The maximum number of recursive calls for the merge algorithm. Defaults to 3.

    Returns:
        List[List[float]]: A list of merged boxes.
    """
    def _apply_merge(bxs):
        final_boxes = []
        amount = len(bxs)
        removed_indexes = [False] * amount
        
        # For each box
        for i in range(amount):
            # If its not removed already
            if removed_indexes[i]: continue
            
            # If it isn't the last one
            if i == amount-1:
                final_boxes.append(bxs[i])
                break
            
            # Get the correspoding coordinates and score
            current_box = bxs[i]
            
            # For each next box (less score)
            for j in range(i+1,amount):
                
                # Get the correspoding coordinates and score
                box_j = bxs[j]
                
                # Check if they should be merged
                merge = get_boxes_min_distance(current_box, box_j) < distance_thresh
                
                # If they should, add the index to the removed and perform merge
                if merge:
                    removed_indexes[j] = True
                    current_box = merge_boxes_by_max_coords(current_box, box_j)
            
            # Add to the final results
            final_boxes.append(current_box)
        return final_boxes
        
    previous_count = len(boxes)
    fbx = boxes
    iteration = 0
    while iteration < max_recursive_calls:
        fbx = _apply_merge(fbx)
        current_count = len(fbx)
        if current_count == previous_count: break
        previous_count = current_count
    return fbx

#%% --- --- --- --- --- --- --- --- ---
# Coordinate computations
def get_box_coordinates_on_new_origin(box, new_left:int, new_top:int, new_right:int, new_bot:int):
    """
    Calculates the new coordinates of a bounding box with respect to a new origin.

    Args:
        box (list | list-like): A list of four integers representing the bounding box coordinates in the format [left, top, right, bottom].
        new_left (int): The new left coordinate of the bounding box.
        new_top (int): The new top coordinate of the bounding box.
        new_right (int): The new right coordinate of the bounding box.
        new_bot (int): The new bottom coordinate of the bounding box.

    Returns:
        list: A list of four integers representing the new bounding box coordinates in the format [left, top, right, bottom].
    """
    # Calculate the coordinates of the bounding box inside the patch
    box_left = np.clip(box[0], new_left, new_right)
    box_top = np.clip(box[1], new_top, new_bot)
    box_right = np.clip(box[2], box_left, new_right)
    box_bottom = np.clip(box[3], box_top, new_bot)

    # Box outside patch
    width_zero = box_left == box_right
    height_zero = box_top == box_bottom
    if width_zero or height_zero:
        return None
    
    new_box_l = box_left - new_left
    new_box_t = box_top - new_top
    new_box_r = box_right - new_left
    new_box_b = box_bottom - new_top
    
    return [new_box_l, new_box_t, new_box_r, new_box_b]

#%% --- --- --- --- --- --- --- --- ---
# Plotting and drawing
def draw_boxes_on_image(image,
                        boxes,
                        labels,
                        label_name_map:dict,
                        label_color_map:dict,
                        font=None,
                        text_offset:tuple[int,int]=(3,0),
                        box_line_width:int=3):
    """
    Draws bounding boxes and labels for objects in an image.

    Args:
        image (PIL.Image): The input image.
        boxes (np.ndarray): An array of bounding boxes, where each box is a 4-element vector [x1, y1, x2, y2].
        labels (np.ndarray): An array of labels corresponding to each box.
        label_name_map (dict): A mapping from label IDs to label names.
        label_color_map (dict): A mapping from label IDs to label colors.
        font (PIL.ImageFont, optional): The font to use for drawing labels. Defaults to None, which loads the default font.
        text_offset (tuple, optional): The offset of the label text from the top-left corner of the bounding box. Defaults to (3, 0).
        box_line_width (int, optional): The width of the bounding box lines. Defaults to 3.

    Returns:
        PIL.Image: The image with bounding boxes and labels drawn on it.
    """
    from PIL import ImageDraw, ImageFont
    
    # Create a new image draw object and font instance
    draw = ImageDraw.Draw(image)
    if font is None:
        font = ImageFont.load_default()

    # Draw the bounding boxes and labels for each object in the image
    for i in range(len(boxes)):
        # Get the coordinates and label id
        x1, y1, x2, y2 = boxes[i].numpy()
        label = labels[i].numpy()

        # Get the label's color and name
        label_color = label_color_map[label]
        label_name = label_name_map[label]
        
        # Draw the box and name
        draw.rectangle((x1, y1, x2, y2), outline=label_color, width=box_line_width)
        draw.text((x1 + text_offset[0], y1 + text_offset[1]), label_name, font=font, fill=label_color)

    return image