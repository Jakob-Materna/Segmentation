#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import glob
import os
import cv2
import time
import warnings

from PIL import Image
from scipy.spatial import cKDTree

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from scipy.ndimage import label, sum as ndimage_sum
from scipy.ndimage import binary_fill_holes

from segment_anything import sam_model_registry, SamPredictor
sys.path.append("..")


def sam_predict_mask(image, input_points, input_labels):
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False,
    )
    
    return masks[0]
    

def postprocess_mask(mask):
    labeled_mask, num_features = label(mask)
    if num_features == 0: 
        return mask
    component_sizes = ndimage_sum(mask, labeled_mask, range(1, num_features + 1))
    largest_component_label = np.argmax(component_sizes) + 1 
    largest_component_mask = labeled_mask == largest_component_label
    clean_mask = binary_fill_holes(largest_component_mask)
    
    return clean_mask


def rotate_image(image, angle):
    """Rotate the image by a given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image
    

def find_longest_rectangle(image):
    """Find the rotated rectangle with the longest long side."""    
    max_long_side = 0
    best_box = None
    best_angle = None

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Test all angles from 0 to 180 degrees
    for angle in range(0, 90, 1):
        rotated_image = rotate_image(image, angle)
        # Find contour
        all_wing_contours, _ = cv2.findContours(rotated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not all_wing_contours:
            continue 

        # Find the largest contour
        largest_contour = max(all_wing_contours, key=cv2.contourArea)

        # Compute the bounding rectangle
        x, y, width, height = cv2.boundingRect(largest_contour)
        long_side = max(width, height)

        # Update the rectangle with the longest long side
        if long_side > max_long_side:
            best_angle = angle
            max_long_side = long_side
            best_box = [
                [x, y],
                [x + width, y],
                [x + width, y + height],
                [x, y + height]
            ]

    # Draw the best rectangle on the rotated image
    if best_box is not None:
        best_box = np.array(best_box, dtype=np.int32)

    return best_angle, best_box
    

def crop_wing(image):
    # Create positive/negative selection points 
    height, width, channels = image.shape
    point_1 = (width/4, height/2)
    point_2 = (width*3/4, height/2)

    neg_select = []
    pos_select = [point_1, point_2]
    
    input_points = np.array(neg_select + pos_select)
    input_labels = np.array([0] * len(neg_select) + [1] * len(pos_select))

    wing_mask = sam_predict_mask(image, input_points, input_labels)
    wing_mask = postprocess_mask(wing_mask)

    # Remove extra dimension
    wing_mask = wing_mask.squeeze()
    
    # Create a white image of the same size as the original image
    white_image = np.ones_like(image) * 255
    
    # Apply the mask to each channel 
    wing_image = np.where(wing_mask[:, :, None], image, white_image)

    # Expand image with white pixels to avoid cropping errors
    expanded_image = cv2.copyMakeBorder(wing_image, 1000, 1000, 1000, 1000, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # Apply thresholding to get a binary image
    _, wing_thresh = cv2.threshold(expanded_image, 250, 255, cv2.THRESH_BINARY)
    
    # Invert the binary image
    wing_inv_thresh = cv2.bitwise_not(wing_thresh)

    # Find the rectangle with the longest long side
    best_angle, best_box = find_longest_rectangle(wing_inv_thresh)

    # Rotate image
    rotated_image = rotate_image(expanded_image, best_angle)

    # Get rectangle properties
    rect = cv2.minAreaRect(best_box)  
    center, size, angle = rect
    height, width = size
    
    # Crop the image
    cropped_image = cv2.getRectSubPix(rotated_image, (int(width)+20, int(height)+20), center)

    # Rotate the cropped image 90 degrees to the right if height > width
    if height > width:
        cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
        
    return cropped_image



if __name__ == "__main__":
    # Start a timer 
    start = time.time()

    # Ignore warnings
    warnings.filterwarnings('ignore')

    # Select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Set up sam predictor checkpoint
    sam_checkpoint = "/home/wsl/bin/segment-anything/checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    predictor = SamPredictor(sam)

    input_dir = "/mnt/c/Projects/Master/Data/Processed/2-LiveWingWingCropsImproved"
    output_dir = "/mnt/c/Projects/Master/Data/Processed/3-LiveWingWingRemovedBackground/"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all jpg files
    jpg_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".JPG") or file.endswith(".jpg"):
                jpg_files.append(os.path.join(root, file))

    # Global warnings counter
    total_warnings = 0

    # Process every file
    total_files = len(jpg_files)
    digits = len(str(total_files))
    for idx, jpg_file_path in enumerate(jpg_files, 1):
        jpg_basename = os.path.basename(jpg_file_path)
        output_file = output_dir + jpg_basename
        relative_jpg_path = jpg_file_path.removeprefix(input_dir)
        print(f"Processing File {idx:0{digits}}/{total_files}:\t{relative_jpg_path}")
        
        # Load the wing image
        image = cv2.imread(jpg_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Process wing
        wing_image = crop_wing(image)

        # Save wing
        wing_image = Image.fromarray(wing_image)
        wing_image.save(output_file)

    # Print Total Warnings
    print(f"\nTotal Warnings: {total_warnings}")

    # End the timer 
    end = time.time()
    duration = end - start

    # Convert to hours, minutes, and seconds
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    # Print the runtime in hh:mm:ss format
    print(f"Script Runtime (hh:mm:ss): {hours:02}:{minutes:02}:{seconds:02}")