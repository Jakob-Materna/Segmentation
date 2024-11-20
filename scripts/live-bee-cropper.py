#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import sys
import glob
import os
import cv2
import time

from PIL import Image
from scipy.ndimage import binary_fill_holes

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# select the sam2 model
sam2_checkpoint = "/home/wsl/bin/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

input_dir = "/mnt/c/Projects/Master/Data/WingImages/Round02/"
output_dir = "/mnt/c/Projects/Master/Data/Testdata/LiveWingScanCrops/"

def find_white_area(image, y_coord, window_size, step_size, density_threshold):
    h, w = image.shape
    max_density = -1
    best_coords = (0, 0)

    # Start searching from the center x-coordinate
    center_x = w // 2

    # Radius-based search around the center x-coordinate
    for radius in range(0, w // 2, step_size):
        # Check positions to the left and right of the center within the current radius
        for dx in range(-radius, radius + 1, step_size):
            for direction in [-1, 1]:  # -1 for left, 1 for right
                x = center_x + dx * direction

                # Ensure the window is within bounds horizontally
                if 0 <= x <= w - window_size and 0 <= y_coord <= h - window_size:
                    # Extract a square window from the image
                    window = image[y_coord:y_coord + window_size, x:x + window_size]

                    # Count the number of white pixels
                    white_pixel_count = np.sum(window >= 120)

                    # Calculate density (fraction of white pixels in the window)
                    density = white_pixel_count / (window_size * window_size)

                    # Track the window with the maximum density of white pixels
                    if density > max_density:
                        max_density = density
                        best_coords = (x, y_coord)

                    # Early termination if a good enough density is found
                    if density >= density_threshold:
                        return best_coords

    return best_coords
    

def identify_label(image, sampling_coords):
    input_point = np.array(sampling_coords)
    input_label = np.array([1] * len(sampling_coords))
    
    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
    )
    mask = masks[0]

    # Fill holes in the mask
    mask = binary_fill_holes(mask).astype(int)
    
    return mask

def crop_from_mask(mask, image):
    # Identification of the label
    # Convert mask to 8-bit single channel
    mask = mask.astype(np.uint8)
    
    # Find contours
    mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Select the largest contour 
    mask_contour = max(mask_contours, key=cv2.contourArea)
    
    # Calculate the minimum area bounding box
    mask_rect = cv2.minAreaRect(mask_contour)
    
    # Get the box points and convert them to integer coordinates
    mask_box_points = cv2.boxPoints(mask_rect)
    mask_box_points = np.intp(mask_box_points)

    # Swap width and height if necessary to make the longer side horizontal
    center, size, angle = mask_rect
    if size[0] < size[1]:
        angle += 90
        size = (size[1], size[0])
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate the entire image to align the rectangle horizontally
    height, width = mask.shape[:2]
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    # Calculate the bounding box of the rotated rectangle in the rotated image
    x, y, w, h = cv2.boundingRect(np.intp(cv2.transform(np.array([mask_box_points]), rotation_matrix))[0])

    # Crop the aligned rectangle with white padding for any areas outside the original image
    cropped_image = rotated_image[y:y+h, x:x+w]
    
    return mask_box_points, cropped_image

    

def crop_label(image, output_dir, jpg_file, verbose=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    label_coords = find_white_area(blurred, y_coord=1000, window_size=200, step_size=200, density_threshold=0.99)
    modified_coord = (label_coords[0] + 100, label_coords[1] + 100)
    coords_list = [label_coords, modified_coord]

    mask = identify_label(image, coords_list)

    mask_box_points, cropped_image = crop_from_mask(mask, image)

    if not verbose:
        # Save cropped label
        os.makedirs(output_dir, exist_ok=True)
        label = Image.fromarray(cropped_image)
        label.save(output_dir + jpg_file)

    else:    
        # Save cropped label
        label_dir = output_dir + "Labels/"
        os.makedirs(label_dir, exist_ok=True)
        label = Image.fromarray(cropped_image)
        label.save(label_dir + jpg_file)

        # Create an image directory
        image_dir = output_dir + "Process/"
        os.makedirs(image_dir, exist_ok=True)
        
        # New 4 channel image (RGBA)
        png_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        
        # Apply the color to each channel (R, G, B)
        for c in range(3):
            png_image[:, :, c] = (mask * (1, 0, 0)[c] * 255).astype(np.uint8)
        
        # Set the alpha channel: 255 where the mask is present, 0 elsewhere
        png_image[:, :, 3] = (mask * 255).astype(np.uint8)
    
        # Draw contours on the image for visualization
        label_image = image.copy()
        cv2.drawContours(label_image, [mask_box_points], 0, (255, 0, 0), 40)

        # Create a 1x3 grid of images
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 20))
        # Show mask
        x_coords, y_coords = zip(*coords_list)
        axes[0].imshow(image)
        axes[0].imshow(png_image, alpha=0.6)
        axes[0].scatter(x_coords, y_coords, c="red", s=20, edgecolor='black')
        axes[0].axis("off")
        # Show rectangle
        axes[1].imshow(label_image)
        axes[1].axis("off")
        # Show cropped image  
        axes[2].imshow(cropped_image)
        axes[2].axis("off")
        plt.savefig(image_dir + jpg_file)
        plt.close()


if __name__ == "__main__":
    # Define directories
    input_dir = "/mnt/c/Projects/Master/Data/WingImages/LiveBees/"
    output_dir = "/mnt/c/Projects/Master/Data/Testdata/LiveWingLabelCrops/"

    # Color palette
    sns_colors = sns.color_palette("hls", 8)

    # Find all jpg files
    jpg_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".JPG") or file.endswith(".jpg"):
                jpg_files.append(os.path.join(root, file))

    # Process every file
    total_files = len(jpg_files)
    digits = len(str(total_files))
    for idx, jpg_file_path in enumerate(jpg_files, 1):
        jpg_basename = os.path.basename(jpg_file_path)
        relative_jpg_path = jpg_file_path.removeprefix(input_dir)
        # output_subdir = output_dir + relative_jpg_path.removesuffix(jpg_basename)
        new_jpg_basename = relative_jpg_path.replace("/", "-")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing file {idx:0{digits}}/{total_files}:\t{relative_jpg_path}")
        image = cv2.imread(jpg_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop_label(image, output_dir, new_jpg_basename, verbose=True)    


