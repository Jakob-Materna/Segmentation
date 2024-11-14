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

# Function to find the closest contour to the horizontal center
def get_contour_closest_to_center(contours, center_x):
    min_distance = float('inf')
    center_contour = None
    
    for contour in contours:
        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            distance = abs(cX - center_x)
            
            # Check if this contour is the closest so far
            if distance < min_distance:
                min_distance = distance
                center_contour = contour

    return center_contour
    
# Color palette
sns_colors = sns.color_palette("hls", 8)

# Ensure the input directory exists
if not os.path.exists(input_dir):
    raise FileNotFoundError(f"Input directory '{input_dir}' was not found.")

sub_dirs = [os.path.join(root, d) for root, dirs, _ in os.walk(input_dir) for d in dirs]

for sub_dir in sub_dirs:
    # Find jpg files
    jpg_files = [file for file in os.listdir(sub_dir) if file.endswith('.JPG')]
    
    for jpg_file in jpg_files:
        output_images = {"Image_1": True,
                         "Image_2": True,
                         "Image_3": True,
                         "Image_4": True,
                         "Image_5": True,
                         "Image_6": True,
                         "Image_7": True,
                         "Image_8": True,
                         "Image_9": True}
        
        input_file = sub_dir + "/" + jpg_file
        print(input_file)
        image = cv2.imread(input_file)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
        # Apply Gaussian Blur
        blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply thresholding to get a binary image
        _, thresh = cv2.threshold(blurred_image, 110, 255, cv2.THRESH_BINARY)
        
        # Invert the binary image
        inv_thresh = cv2.bitwise_not(thresh)
    
        # Find contour
        marker_contours, _ = cv2.findContours(inv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # large_marker_contours = [cnt for cnt in marker_contours if (45000 > cv2.contourArea(cnt) > 20000)]
        large_marker_contours = [cnt for cnt in marker_contours if (50000 > cv2.contourArea(cnt) > 10000)]
        # contour = large_contours[0]
    
        # Draw contours on the image for visualization
        wing_contour_image_1 = image.copy()
        cv2.drawContours(wing_contour_image_1, marker_contours, -1, (255, 0, 0), 20)
        
        # Draw contours on the image for visualization
        wing_contour_image_2 = image.copy()
            
        cv2.drawContours(wing_contour_image_2, large_marker_contours, -1, (255, 0, 0), 20)
    
        # Create a copy of the image to draw on
        marker_contours = image.copy()
    
        # List of identified lines
        markers = []
    
        # length of the line 
        marker_length = []
        
        # Loop over contours and find the minimum-area bounding rectangle
        for marker_contour in large_marker_contours:
            # Get the minimum-area rectangle for each contour
            marker_rect = cv2.minAreaRect(marker_contour)
    
            # Extract the width and height of the rectangle
            (center_x, center_y), (height, width), angle = marker_rect
            
            # Calculate the area of the rectangle
            # area = width * height
            # print(f"width: {int(width)}\theight: {int(height)}\tarea:{int(area)}")
            
            # Extract the box points and convert them to integers
            marker_box = cv2.boxPoints(marker_rect)
            marker_box = np.intp(marker_box)

            
            # if (650 > height > 550) and (150 > width > 50):
            if (700 > height > 500) and (150 > width > 10):
                print(f"Length of identified marker: {height}")
                markers.append(marker_contour)
                marker_length.append(height)
                # Draw the rectangle on the output image
                cv2.drawContours(marker_contours, [marker_box], 0, (0, 0, 255), 20)
            else:
                cv2.drawContours(marker_contours, [marker_box], 0, (255, 0, 0), 20)
    
        if markers:
            if len(markers) > 2:
                print("\tWARNING! More than two makers identified!")
            # TODO: Maybe take the average? 
            marker = markers[0]
            marker_length = max(marker_length)
            print(f"Length of longest identified marker: {marker_length}")
            
            # Find centroid
            M = cv2.moments(marker)
            if not M['m00'] != 0:
                print("No centroid found!")
                exit()
                
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroid = [cx, cy]
            sampling_point = [cx - 100, cy]
        
            input_point = np.array([sampling_point])
            input_label = np.array([1])
            
            predictor.set_image(image)
        
            masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
            )
            mask = masks[0]
        
            # Fill holes in the mask
            mask = binary_fill_holes(mask).astype(int)
            
            # New 4 channel image (RGBA)
            png_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            
            # Apply the color to each channel (R, G, B)
            for c in range(3):  # Loop over RGB channels
                png_image[:, :, c] = (mask * sns_colors[2][c] * 255).astype(np.uint8)  # Scale colors to [0, 255]
            
            # Set the alpha channel: 255 where the mask is present, 0 elsewhere
            png_image[:, :, 3] = (mask * 255).astype(np.uint8)
        
            
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
        
            # Draw contours on the image for visualization
            label_image = image.copy()
            cv2.drawContours(label_image, [mask_box_points], 0, (255, 0, 0), 40)
            
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
        
            # Make a copy of the cropped image
            modified_crop = cropped_image.copy()
            
            # Draw an inner white border
            cv2.rectangle(modified_crop, (0, 0), (modified_crop.shape[1], modified_crop.shape[0]), (255, 255, 255), thickness=100)
            
            # Convert to grayscale
            gray_cropped = cv2.cvtColor(modified_crop, cv2.COLOR_BGR2GRAY)
        
            # Apply Gaussian Blur
            blurred_cropped = cv2.GaussianBlur(gray_cropped, (5, 5), 0)
        
            # Apply thresholding to create a binary image
            _, binary_cropped = cv2.threshold(blurred_cropped, 128, 255, cv2.THRESH_BINARY)
        
            # Invert the binary image
            binary_cropped = cv2.bitwise_not(binary_cropped)
        
            # Fill holes in the binary image 
            # binary_cropped = binary_fill_holes(binary_cropped)
        
            # Find contours in the thresholded image
            wing_contours, _ = cv2.findContours(binary_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter out small contours
            large_wing_contours = [cnt for cnt in wing_contours if cv2.contourArea(cnt) > 100000]
            
            # Visualize the large contours
            wing_contour_image = cropped_image.copy()
            cv2.drawContours(wing_contour_image, large_wing_contours, -1, (255, 0, 0), 20)

            if len(large_wing_contours) > 1:
                # Identify central contour
                label_height, label_width = cropped_image.shape[:2]
                label_center_x = label_width // 2  
                wing_contour = get_contour_closest_to_center(large_wing_contours, label_center_x)

            if len(large_wing_contours) == 1:    
                # Get the minimum-area rectangle for the wing contour
                wing_contour = large_wing_contours[0]
                print(f"Visible wing area: {cv2.contourArea(wing_contour)}")
                wing_rect = cv2.minAreaRect(wing_contour)
            
                # Extract the width and height of the rectangle
                (center_x, center_y), (height, width), angle = wing_rect
                
                # Extract the box points and convert them to integers
                wing_box_points = cv2.boxPoints(wing_rect)
                wing_box_points = np.intp(wing_box_points)
            
                # Draw contours on the image for visualization
                wing_image = cropped_image.copy()
                cv2.drawContours(wing_image, [wing_box_points], 0, (255, 0, 0), 20)
            
                # Swap width and height if necessary to make the longer side horizontal
                center, size, angle = wing_rect
                if size[0] < size[1]:
                    angle += 90
                    size = (size[1], size[0])
            
                # Get the rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Rotate the entire image to align the rectangle horizontally
                height, width = cropped_image.shape[:2]
                rotated_wing_image = cv2.warpAffine(cropped_image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
            
                # Calculate the bounding box of the rotated rectangle in the rotated image
                x, y, w, h = cv2.boundingRect(np.intp(cv2.transform(np.array([wing_box_points]), rotation_matrix))[0])
            
                # Crop the aligned rectangle with white padding for any areas outside the original image
                t = 20
                cropped_wing_image = rotated_wing_image[y-t:y+h+t, x-t:x+w+t]
            
            else:
                print(f"\tWARINING! No matching wing contours: {jpg_file}")
                output_images = {"Image_1": True,
                         "Image_2": True,
                         "Image_3": True,
                         "Image_4": True,
                         "Image_5": True,
                         "Image_6": True,
                         "Image_7": True,
                         "Image_8": True,
                         "Image_9": False}
    
        else:
            print(f"\tWARINING! No matching marker contours: {jpg_file}")
            output_images = {"Image_1": True,
                             "Image_2": True,
                             "Image_3": True,
                             "Image_4": False,
                             "Image_5": False,
                             "Image_6": False,
                             "Image_7": False,
                             "Image_8": False,
                             "Image_9": False}
        # Create a 2x2 grid of images
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(30, 20))
        
        # Display each image in a separate subplot
        if output_images["Image_1"]:
            axes[0, 0].imshow(thresh, cmap="gray")
            axes[0, 0].set_title("Threshold Image", size=25)
        axes[0, 0].axis('off')
        if output_images["Image_2"]:
            axes[0, 1].imshow(wing_contour_image_1)
            axes[0, 1].set_title("All Contours", size=25)
        axes[0, 1].axis('off')
        if output_images["Image_3"]:
            axes[0, 2].imshow(wing_contour_image_2)
            axes[0, 2].set_title("Filtered Contours", size=25)
        axes[0, 2].axis('off')
        if output_images["Image_4"]:
            axes[1, 0].imshow(marker_contours)
            axes[1, 0].set_title("Filtered Rectangle Contours", size=25)
            axes[1, 0].text(50, 350, f"length of first rectangle: {round(marker_length, 2)} pixels", color="blue", size=20)
        axes[1, 0].axis('off')
        if output_images["Image_5"]:
            axes[1, 1].imshow(image)
            axes[1, 1].imshow(png_image, alpha=0.6)
            axes[1, 1].scatter(centroid[0], centroid[1], color=sns_colors[0], s=40, label='Exclude', edgecolor='black')
            axes[1, 1].scatter(sampling_point[0], sampling_point[1], color=sns_colors[2], s=40, label='Exclude', edgecolor='black')
            axes[1, 1].set_title("Centroid, Sampling Point and SAM Mask", size=25)
        axes[1, 1].axis('off')
        if output_images["Image_6"]:
            axes[1, 2].imshow(label_image, cmap="gray")
            axes[1, 2].set_title("Rectanglular approximation of the Mask", size=25)
        axes[1, 2].axis('off')
        if output_images["Image_7"]:
            axes[2, 0].imshow(cropped_image)
            axes[2, 0].set_title("Cropped Label", size=25)
        axes[2, 0].axis('off')
        if output_images["Image_8"]:
            axes[2, 1].imshow(wing_contour_image)
            axes[2, 1].set_title("Wing Contour", size=25)
        axes[2, 1].axis('off')
        if output_images["Image_9"]:
            axes[2, 2].imshow(cropped_wing_image)
            axes[2, 2].set_title("Cropped Wing", size=25)
        axes[2, 2].axis('off')
        # Show the plot
        all_true = all(output_images.values())
        base_name = jpg_file.removesuffix(".JPG")
        if all_true:
            image_dir = output_dir + "Successful/"
        else:
            image_dir = output_dir + "Failed/"
        os.makedirs(image_dir, exist_ok=True)
        plt.savefig(image_dir + jpg_file)
        # plt.show()
        plt.close()
        
        if output_images["Image_9"]:
            wing_dir = output_dir + "Wings/"
            os.makedirs(wing_dir, exist_ok=True)
            wing = Image.fromarray(cropped_wing_image)
            wing.save(wing_dir + jpg_file)
            
        cv2.destroyAllWindows()