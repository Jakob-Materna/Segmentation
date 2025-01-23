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


def remove_points_near_border(points, contour, border_dist_threshold):
    filtered_points = []

    # Iterate over all points
    for point in points:
        # Check the distance of the point to the contour
        dist_to_contour = cv2.pointPolygonTest(contour, (point[0], point[1]), True)
        
        # Keep the point if it's farther from the border than the threshold
        if dist_to_contour >= border_dist_threshold:
            filtered_points.append(point)
    
    return np.array(filtered_points)

    
def find_black_area(image, window_size):
    h, w = image.shape
    max_density = -1
    best_coords = (0, 0)

    # Slide the window over the image
    for y in range(0, h - window_size[1] + 1, 1):
        for x in range(0, w - window_size[0] + 1, 1):
            # Extract the window from the image
            window = image[y:y + window_size[1], x:x + window_size[0]]

            # Count the number of black pixels (assuming black pixels are 0)
            black_pixel_count = np.sum(window == 0)

            # Track the window with the maximum number of black pixels
            if black_pixel_count > max_density:
                max_density = black_pixel_count
                best_coords = (x, y)

    return best_coords, max_density
        

def remove_background(wing):
    # Show image
    # plt.figure(figsize=(5, 5))
    # plt.imshow(cv2.cvtColor(wing, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
    
    # expanded_image = cv2.copyMakeBorder(wing, 1000, 1000, 1000, 1000, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    expanded_image = wing
    # Convert the image to grayscale
    gray = cv2.cvtColor(expanded_image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Invert the binary image
    inv_thresh = cv2.bitwise_not(thresh)
    
    # Show image
    # plt.figure(figsize=(5, 5))
    # plt.imshow(inv_thresh, cmap="gray")
    # plt.axis('off')
    # plt.show()
    
    # Find contour
    contours, _ = cv2.findContours(inv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
    contour = large_contours[0]
    
    # Draw contours on the image for visualization
    wing_contour_image = expanded_image.copy()
    cv2.drawContours(wing_contour_image, large_contours, -1, (0, 0, 255), 5)
    
    # Show image
    # plt.figure(figsize=(5, 5))
    # plt.imshow(cv2.cvtColor(wing_contour_image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
    
    # Get bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Create a dense grid of points within the bounding box
    distance = 100  
    height, width, channels = expanded_image.shape  
    
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
    
    # inside_points = np.array(inside_points)
    
    filtered_points = remove_points_near_border(inside_points, contour, 25)
    # Find the coordinates of the area with the highest density of black pixels
    # best_coords, max_density = find_black_area(gray, window_size)

    
    window_size = (5, 5)
    best_coords, max_density = find_black_area(gray, window_size)
    
    # Plot the contour and the selected points
    # plt.figure(figsize=(5, 5))
    # plt.imshow(cv2.cvtColor(wing_contour_image, cv2.COLOR_BGR2RGB))
    # plt.scatter(best_coords[0], best_coords[1], c="green", s=10)
    # plt.scatter(filtered_points[:, 0], filtered_points[:, 1], c="red", s=5)
    # plt.axis('off')
    # plt.show()
    
    filtered_points = np.vstack([filtered_points, best_coords])

    # Sam background removal
    # Convert the points list to a numpy array
    image_points = np.array(filtered_points)
    image_labels = np.array([1] * len(filtered_points))
    
    predictor.set_image(expanded_image)
    
    mask, score, _ = predictor.predict(
        point_coords=image_points,
        point_labels=image_labels,
        multimask_output=False,
    )
    sorted_ind = np.argsort(score)[::-1]
    mask = mask[sorted_ind]
    score = score[sorted_ind]
    
    # show_masks(expanded_image, masks, scores, point_coords=point, input_labels=label, borders=True)
    
    # Remove extra dimension
    mask = mask.squeeze()
    
    # Create a white image of the same size as the original image
    white_image = np.ones_like(expanded_image) * 255
    
    # Apply the mask to each channel (no extra dimension added)
    wing_image = np.where(mask[:, :, None], expanded_image, white_image)
    """
    # Show image
    plt.figure(figsize=(20, 20))
    plt.imshow(cv2.cvtColor(wing_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    """
    return wing_image

def crop_wing(wing_image):
    expanded_image = cv2.copyMakeBorder(wing_image, 1000, 1000, 1000, 1000, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(expanded_image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Invert the binary image
    inv_thresh = cv2.bitwise_not(thresh)
    """
    # Show image
    plt.figure(figsize=(5, 5))
    plt.imshow(inv_thresh, cmap="gray")
    plt.axis('off')
    plt.show()
    """
    # Find contour
    contours, _ = cv2.findContours(inv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]
    contour = large_contours[0]
    
    # Draw contours on the image for visualization
    wing_contour_image = expanded_image.copy()
    cv2.drawContours(wing_contour_image, large_contours, -1, (0, 0, 255), 5)
    """
    # Show image
    plt.figure(figsize=(5, 5))
    plt.imshow(wing_contour_image)
    plt.axis('off')
    plt.show()
    """
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(contour)
    
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
        angle -= 90  # Rotate to make the longest side horizontal
    
    # Get the rotation matrix to rotate the image around the rectangle's center
    rotation_matrix = cv2.getRotationMatrix2D(box_center, angle, 1.0)
    
    # Rotate the entire image
    rotated_image = cv2.warpAffine(expanded_image, rotation_matrix, (expanded_image.shape[1], expanded_image.shape[0]))
    
    # Convert the center and size to integers
    box_center = (int(box_center[0]), int(box_center[1]))
    width, height = int(width), int(height)
    
    # Crop the aligned rectangle from the rotated image
    cropped_image = cv2.getRectSubPix(rotated_image, (width+20, height+20), box_center)
    """
    # Show image
    plt.figure(figsize=(5, 5))
    plt.imshow(contour_image)
    plt.axis('on')
    plt.show()

    # Show image
    plt.figure(figsize=(5, 5))
    plt.imshow(cropped_image)
    plt.axis('on')
    plt.show()
    """
    return(cropped_image)


if __name__ == "__main__":
    # Start a timer 
    start = time.time()

    # Ignore warnings
    warnings.filterwarnings('ignore')
    
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    sam2_checkpoint = "/home/wsl/bin/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    
    predictor = SAM2ImagePredictor(sam2_model)

    input_dir = "/mnt/c/Projects/Master/Data/ProcessedWings/WingScanCrops"
    output_dir = "/mnt/c/Projects/Master/Data/Segmented/"
        
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' was not found.")
    
    # Create the output directory
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        print(f"Output directory already exists.")
        # exit()
        
    # List all directories in the specified directory
    all_directories = [entry for entry in os.listdir(input_dir)]
    
    for dirname in all_directories:
        if not "Hive" in dirname:
            print(f"Skipping directory: {dirname}")
            continue
            
        input_subdir = input_dir + "/" + dirname + "/"
        output_subdir = output_dir + "/" + dirname + "/"
        
        # Create the output directory
        try:
            os.makedirs(output_subdir)
        except FileExistsError:
            print(f"Output directory already exists. Skipping directory: {dirname} ")
            continue
            
        print(f"Processing directory: {dirname}")    
        # Find jpg files
        jpg_files = [file for file in os.listdir(input_subdir) if file.endswith('.jpg')]
        for jpg_file in jpg_files:
            input_file = input_subdir + jpg_file
            output_file = output_subdir + jpg_file
            wing = Image.open(input_file)
            wing = np.array(wing.convert("RGB"))
            
            height, width, channels = wing.shape
            if height >= 1000 or width >= 1000:
                print(f"\tWarning: Image '{jpg_file}' is too large " \
                f"(width: {width}, height: {height}) and has been skipped.")
                continue
                
            try:
                wing = remove_background(wing)
                cropped_image = crop_wing(wing)
            except Exception: 
                print(f"\tWarning: Image '{jpg_file}' could not be processed.")
                continue
            # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image = Image.fromarray(cropped_image)
            if "Left" in jpg_file:
                cropped_image = cropped_image.transpose(method=Image.FLIP_LEFT_RIGHT)
                # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                # cropped_image = Image.fromarray(cropped_image)
        
            # Show image
            # plt.figure(figsize=(5, 5))
            # plt.imshow(cropped_image)
            # plt.axis('on')
            # plt.show()
            
            cropped_image.save(output_file)

    # Print script runtime 
    end = time.time()
    
    # Calculate the elapsed time
    duration = end - start
    
    # Convert to hours, minutes, and seconds
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    # Print the runtime in hh:mm:ss format
    print(f"Script runtime (hh:mm:ss): {hours:02}:{minutes:02}:{seconds:02}")
