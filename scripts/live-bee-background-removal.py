#!/usr/bin/env python3

import numpy as np
import torch
import sys
import os
import cv2
import time
import warnings

from PIL import Image

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


def crop_wing(image):
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

    # Expand the wing image
    expanded_image = cv2.copyMakeBorder(wing_image, 1000, 1000, 1000, 1000, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # Apply thresholding to get a binary image
    _, wing_thresh = cv2.threshold(expanded_image, 250, 255, cv2.THRESH_BINARY)
    
    # Invert the binary image
    wing_inv_thresh = cv2.bitwise_not(wing_thresh)
    wing_inv_thresh = cv2.cvtColor(wing_inv_thresh, cv2.COLOR_RGB2GRAY)
    
    # Find contour
    all_wing_contours, _ = cv2.findContours(wing_inv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    largest_contour = max(all_wing_contours, key=cv2.contourArea)

    # Get the minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    
    # Get the four points of the rectangle
    box = cv2.boxPoints(rect)
    
    # Convert the points to integers
    box = np.intp(box)

    # Draw the rotated rectangle
    contour_image = expanded_image.copy()
    cv2.drawContours(contour_image, [box], 0, (0, 0, 255), 5)
    
    # Get the rectangle's center, size (width, height), and angle
    box_center, box_size, angle = rect
    
    # Ensure width is the longest side (width > height)
    width, height = box_size
    if height > width:
        width, height = height, width
        angle -= 90 
    
    # Get the rotation matrix to rotate the image around the rectangle's center
    rotation_matrix = cv2.getRotationMatrix2D(box_center, angle, 1.0)
    
    # Rotate the entire image
    rotated_image = cv2.warpAffine(expanded_image, rotation_matrix, (expanded_image.shape[1], expanded_image.shape[0]))
    
    # Convert the center and size to integers
    box_center = (int(box_center[0]), int(box_center[1]))
    width, height = int(width), int(height)
    
    # Crop the aligned rectangle from the rotated image
    cropped_image = cv2.getRectSubPix(rotated_image, (width+50, height+50), box_center)

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

    input_dir = "/mnt/c/Projects/Master/Data/Processed/3-LiveWingCropsImproved"
    output_dir = "/mnt/c/Projects/Master/Data/Processed/4-LiveWingCropsRemovedBackground/"

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
        relative_jpg_path = relative_jpg_path.removeprefix("/")

        # Skip if the file exists
        if os.path.exists(output_file):
            print(f"Output already exists. Skipping File {idx:0{digits}}/{total_files}:\t{relative_jpg_path}")
            continue
        
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
