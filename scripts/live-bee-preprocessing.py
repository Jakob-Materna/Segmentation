#!/usr/bin/env python3

import seaborn as sns
import pandas as pd
import numpy as np
import statistics
import os
import cv2
import time

from PIL import Image
from scipy.ndimage import binary_fill_holes


def get_lower_y(marker_box):
    # Find the y-coordinates of all four corners of the rectangle
    y_coords = marker_box[:, 1]
    return np.mean(y_coords) 
    

def find_markers(inv_thresh, all_contours, image):
    large_marker_contours = [cnt for cnt in all_contours if (100000 > cv2.contourArea(cnt) > 20000)]
        
    # Create a copy of the image to draw on
    marker_contours = image.copy()

    # length of the line 
    marker_length = []
    
    # Loop over contours 
    for marker_contour in large_marker_contours:
        # Get the minimum-area rectangle for each contour
        marker_rect = cv2.minAreaRect(marker_contour)

        # Extract the width and height of the rectangle
        (center_x, center_y), (length, width), angle = marker_rect
        
        # Ensure length is the longer side
        if length < width:
            length, width = width, length
            # Adjust angle when swapping length and width
            angle = angle + 90  

        # Normalize the angle to 90 degree range (all angles are now between 0 and 90 degrees)
        angle = angle % 90

        # Extract the box points and convert them to integers
        marker_box = cv2.boxPoints(marker_rect)
        marker_box = np.intp(marker_box)
        
        # Filter rectangles
        if (800 > length > 500) and (150 > width > 10) and ((5 > angle > 0) or (90 > angle > 85)):
            marker_length.append(length)
            # Draw the rectangle on the output image
            cv2.drawContours(marker_contours, [marker_box], 0, (0, 0, 255), 20)
        else:
            cv2.drawContours(marker_contours, [marker_box], 0, (255, 0, 0), 20)

    return marker_length, marker_contours


def find_wing(inv_thresh, all_contours, image):
    large_wing_contours = [cnt for cnt in all_contours if (1000000 > cv2.contourArea(cnt) > 100000)]

    # Create a copy of the image to draw on
    wing_contours = image.copy()

    # Save wing contours
    wing_boxes = []
    wing_rects = []
    # Store pairs of (wing_box, wing_rect)
    wing_data = [] 
    
    # Loop over contours and find the minimum-area bounding rectangle
    for wing_contour in large_wing_contours:
        # Get the minimum-area rectangle for each contour
        wing_rect = cv2.minAreaRect(wing_contour)
        wing_rects.append(wing_rect)
        
        # Extract the box points and convert them to integers
        wing_box = cv2.boxPoints(wing_rect)
        wing_box = np.intp(wing_box)
        wing_boxes.append(wing_box)
        
        # Store the pair
        wing_data.append((wing_box, wing_rect))
    
    # If no valid contours are found, return early
    if len(large_wing_contours) == 0:
        return None, None, wing_boxes, wing_contours
    
    # Identify the lower rectangle (box with the lowest center y-coordinate)
    def get_lower_y(wing_box):
        return np.mean(wing_box[:, 1])  # Average y-coordinate of box points
    
    # Find the pair for the lower rectangle
    lower_rectangle_box, lower_rectangle_rect = max(wing_data, key=lambda data: get_lower_y(data[0]))
    
    # Visualization
    for wing_box in wing_boxes:
        cv2.drawContours(wing_contours, [wing_box], -1, (255, 0, 0), 20)  # Draw all boxes in blue
    
    # Highlight the lower rectangle in red
    cv2.drawContours(wing_contours, [lower_rectangle_box], -1, (0, 0, 255), 20)
    
    # Return the lowest rectangle box, its matching rect, and the contour image
    return lower_rectangle_box, lower_rectangle_rect, wing_boxes, wing_contours


def crop_wing(wing_box, wing_rect, image):
    # Extract the width and height of the rectangle
    center_rect, (height_rect, width_rect), angle_rect = wing_rect

    # Swap width and height if necessary to make the longer side horizontal
    if height_rect < width_rect:
        angle_rect += 90
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center_rect, angle_rect, 1.0)
    
    # Rotate the entire image to align the rectangle horizontally
    image_height, image_width = image.shape[:2]
    rotated_wing_image = cv2.warpAffine(image, rotation_matrix, (image_width, image_height), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    
    # Calculate the bounding box of the rotated rectangle in the rotated image
    x, y, w, h = cv2.boundingRect(np.intp(cv2.transform(np.array([wing_box]), rotation_matrix))[0])
    
    # Crop the aligned rectangle with white padding for any areas outside the original image
    t = 100
    cropped_wing_image = rotated_wing_image[y-t:y+h+t, x-t:x+w+t]

    return cropped_wing_image


def preprocessing_main(gray, image, output_file):
    global total_warnings

    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)

    all_marker_lengths = []
    all_wing_crops = []
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
        label_contour_image = cv2.cvtColor(inv_thresh, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(label_contour_image, all_contours, -1, (255, 0, 0), 20)
        
        # Find markers
        marker_length, marker_contours = find_markers(inv_thresh, all_contours, image)   
        all_marker_lengths += marker_length
        
        # Find wing
        wing_box, wing_rect, wing_rects, wing_contours = find_wing(inv_thresh, all_contours, image)

        # Crop wing
        if wing_rect is not None:
            cropped_wing_image = crop_wing(wing_box, wing_rect, image)
            if cropped_wing_image.any():
                all_wing_crops.append(cropped_wing_image)
            
        # Decrease threshold until 0 is reached 
        threshold -= 10

    if len(all_wing_crops) >= 1:
        wing = all_wing_crops[len(all_wing_crops) // 2]        
        wing = Image.fromarray(wing)
        wing.save(output_file)
    else:
        total_warnings += 1
        print("\tWARNING: No Wings Identified!")
        
    if len(all_marker_lengths) >= 1:
        # Return the mean marker length
        mean_length = statistics.mean(all_marker_lengths)
        return mean_length
    else:
        total_warnings += 1
        print("\tWARNING: No Markers Identified!")
        return None


if __name__ == "__main__":
    # Start a timer 
    start = time.time()

    # Define directories
    input_dir = "/mnt/c/Projects/Master/Data/Processed/1-LiveWingLabelCrops/Labels"
    output_dir = "/mnt/c/Projects/Master/Data/Processed/2-LiveWingWingCrops/"

    # Color palette
    sns_colors = sns.color_palette("hls", 8)

    # Find all jpg files
    jpg_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".JPG") or file.endswith(".jpg"):
                jpg_files.append(os.path.join(root, file))

    # Create output directory
    output_subdir = output_dir + "/Wings/"
    os.makedirs(output_subdir, exist_ok=True)

    # Empty list for marker length table
    markers = []

    # Global warnings counter
    total_warnings = 0

    # Process every file
    total_files = len(jpg_files)
    digits = len(str(total_files))
    for idx, jpg_file_path in enumerate(jpg_files, 1):
        jpg_basename = os.path.basename(jpg_file_path)
        output_file = output_subdir + jpg_basename
        relative_jpg_path = jpg_file_path.removeprefix(input_dir)
        print(f"Processing File {idx:0{digits}}/{total_files}:\t{relative_jpg_path}")
        
        # Load image
        image = cv2.imread(jpg_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crop pixels from each side
        padding = 50
        height, width, _ = image.shape
        image = image[padding:height-padding, padding:width-padding]

        # Grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Run the main function
        marker_length = preprocessing_main(gray, image, output_file)

        # Append to results
        markers.append({"jpg_name": jpg_basename, "mean_marker_length_in_pixels": marker_length})

    # Save to Excel
    df = pd.DataFrame(markers)
    output_excel_path = output_dir + "marker_lengths.xlsx"
    df.to_excel(output_excel_path, index=False)

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