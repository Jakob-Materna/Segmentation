#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import cv2
import time

from PIL import Image

from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes

from segment_anything import sam_model_registry, SamPredictor


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

sam_checkpoint = "/home/wsl/bin/segment-anything/checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)


def remove_points_near_border(points, contour, border_dist_threshold):
    filtered_points = []

    # Iterate over all points
    for point in points:
        # Check the distance of the point to the contour
        dist_to_contour = cv2.pointPolygonTest(contour, (point[0], point[1]), True)
        
        # Keep the point if it's farther from the border than the threshold
        if dist_to_contour >= border_dist_threshold:
            filtered_points.append(point)
    
    return filtered_points


def find_black_area(image, window_size):
    h, w = image.shape
    max_density = -1
    best_coords = (0, 0)

    # Slide the window over the image
    for y in range(0, h - window_size[1] + 1, 1):
        for x in range(0, w - window_size[0] + 1, 1):
            # Extract the window from the image
            window = image[y:y + window_size[1], x:x + window_size[0]]

            # Count the number of black pixels
            black_pixel_count = np.sum(window == 0)

            # Track the window with the maximum number of black pixels
            if black_pixel_count > max_density:
                max_density = black_pixel_count
                best_coords = (x, y)

    return best_coords, max_density


def remove_background(wing_image):
    # Grayscale image
    gray = cv2.cvtColor(wing_image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
    
    global total_warnings
    contour_images = []
    wing_contours = []
    
    threshold = 250
    while threshold >= 0:
        # Apply thresholding to get a binary image
        _, thresh = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY)
    
        # Invert the binary image
        inv_thresh = cv2.bitwise_not(thresh)
        
        # Fill holes in the mask
        inv_thresh = binary_fill_holes(inv_thresh).astype(np.uint8) 

        # Scale to match the binary image format
        inv_thresh = inv_thresh * 255

        # Find contour
        all_contours, _ = cv2.findContours(inv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the image for visualization
        wing_contour_image_1 = wing_image.copy()
        cv2.drawContours(wing_contour_image_1, all_contours, -1, (255, 0, 0), 20)

        # Ensure there are contours before proceeding
        wing_contour_image_2 = wing_image.copy()
        if all_contours:
            # Find the largest contour by area
            largest_contour = max(all_contours, key=cv2.contourArea)
        
            # Optional: Get the area of the largest contour (for verification or further use)
            largest_area = cv2.contourArea(largest_contour)
    
            # Calculate the total image area
            image_area = inv_thresh.shape[0] * inv_thresh.shape[1]
        
            # Calculate the percentage
            percentage_area = (largest_area / image_area) * 100

            # print(f"Contour area: {percentage_area}")
            if (50 > percentage_area > 20):
                cv2.drawContours(wing_contour_image_2, largest_contour, -1, (0, 0, 255), 10)
                contour_images.append(wing_contour_image_2)
                wing_contours.append(largest_contour)
            else:
                cv2.drawContours(wing_contour_image_2, largest_contour, -1, (255, 0, 0), 10)

        threshold -= 5

    if len(contour_images) == 0:
        print("No Contours Identified!")
        total_warnings += 1
        return None
        
    wing = contour_images[len(contour_images) // 2]
    contour = wing_contours[len(wing_contours) // 2]

    # Get bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Create a dense grid of points within the bounding box
    distance = 200  
    height, width, channels = wing_image.shape  
    
    # Create x and y coordinates
    x_coords = np.arange(0, width, distance)
    y_coords = np.arange(0, height, distance)
    
    # Create a meshgrid from the x and y coordinates
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    
    # Stack the x and y coordinates into a single array of points
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # Convert the NumPy array to a list of tuples with standard integers
    grid_points = [(int(x), int(y)) for x, y in grid_points]
    
    inside_points = []
    
    # Check if points are inside the contour
    for point in grid_points:
        if cv2.pointPolygonTest(contour, (point[0], point[1]), False) >= 0:
            inside_points.append(point)

    _, thresh_50 = cv2.threshold(blurred_image, 50, 255, cv2.THRESH_BINARY)
    
    window_size = (20, 20)
    best_coords, max_density = find_black_area(thresh_50, window_size)
    if best_coords != (0, 0):
        inside_points.append(best_coords)
        
    filtered_points = np.array(remove_points_near_border(inside_points, contour, 25))

    # Convert the points list to a numpy array
    image_points = np.array(filtered_points)
    image_labels = np.array([1] * len(filtered_points))

    predictor.set_image(wing_image)
    
    mask, score, _ = predictor.predict(
        point_coords=image_points,
        point_labels=image_labels,
        multimask_output=False,
    )
    sorted_ind = np.argsort(score)[::-1]
    mask = mask[sorted_ind]
    score = score[sorted_ind]
    
    # Remove extra dimension
    mask = mask.squeeze()

    # Fill holes in the mask
    filled_mask = binary_fill_holes(mask).astype(int)
    
    # Create a white image of the same size as the original image
    white_image = np.ones_like(wing_image) * 255
    
    # Apply the mask to each channel (no extra dimension added)
    wing_image = np.where(filled_mask[:, :, None], wing_image, white_image)
    
    return wing_image


if __name__ == "__main__":
    # Start a timer 
    start = time.time()

    input_dir = "/mnt/c/Projects/Master/Data/Processed/2-LiveWingWingCropsImproved/"
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
        
        # Process image
        image = cv2.imread(jpg_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        wing_image = remove_background(image)

        if wing_image is not None:
            wing_image = Image.fromarray(wing_image)
            wing_image.save(output_file)

    # Print Total Warnings
    print(f"\nTotal Warnings Across All Files: {total_warnings}")

    # End the timer 
    end = time.time()
    duration = end - start

    # Convert to hours, minutes, and seconds
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    # Print the runtime in hh:mm:ss format
    print(f"Script Runtime (hh:mm:ss): {hours:02}:{minutes:02}:{seconds:02}")
