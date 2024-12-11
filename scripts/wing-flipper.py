#!/usr/bin/env python3

import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import time
import cv2
import os


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

    return best_coords


if __name__ == "__main__":
    # Start a timer 
    start = time.time()

    # Define directories
    input_dir = "/mnt/c/Projects/Master/Data/Processed/3-LiveWingCropsImproved/"

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
        output_file = input_dir + jpg_basename
        relative_jpg_path = jpg_file_path.removeprefix(input_dir)
        print(f"Processing File {idx:0{digits}}/{total_files}:\t{relative_jpg_path}")

        # Load image
        image = cv2.imread(jpg_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian Blur
        blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)

        # Find a threshold with less than 3% black area
        threshold = 60
        while threshold >= 5:
            # Apply thresholding to get a binary image 
            _, thresh = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY)
            
            # Calculate the percentage of black pixels
            total_pixels = thresh.size
            black_pixels = np.count_nonzero(thresh == 0)
            black_percentage = (black_pixels / total_pixels) * 100

            # Use this threshold if less than 3% of pixels are black
            if black_percentage < 3:
                break
                
            # Decrease threshold until less than 3% of pixels are black
            threshold -= 5
        
        window_size = (50, 50)
        cords = find_black_area(thresh, window_size)

        if cords[1] > image.shape[0]/2:
            flipped_image = cv2.flip(image, 0)
            flipped_image = Image.fromarray(flipped_image)
            flipped_image.save(output_file)

    # End the timer 
    end = time.time()
    duration = end - start

    # Convert to hours, minutes, and seconds
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    # Print the runtime in hh:mm:ss format
    print(f"Script Runtime (hh:mm:ss): {hours:02}:{minutes:02}:{seconds:02}")
