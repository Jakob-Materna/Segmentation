import pytesseract
import numpy as np
import sys
import os
import cv2

# import scipy
import matplotlib.pyplot as plt

from PIL import Image
from scipy.ndimage import rotate

# Path to pytesseract-ocr executable. See readme for installation.
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
custom_config='--psm 10 --oem 3 -c tessedit_char_whitelist=wW0123456789'
# custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
# custom_config = '--psm 6'

# Define directories
input_dir = "/mnt/c/Projects/Master/Testdata/Scans/"
output_dir = "/mnt/c/Projects/Master/Testdata/Temp/"


def generate_label_mask(scan):    
    # Set red and green channels to zero to ignore red marker
    # scan_b = scan.copy()
    # scan_b[:, :, 0] = 0
    # scan_b[:, :, 1] = 0

    # Grayscale image
    gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Create a new image to show the contours
    contour_img = scan.copy()
    
    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask
    mask = np.zeros(contour_img.shape[:2], dtype=np.int32)  
    
    # Filter the contours and label them in the mask
    label = 1 
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the contour has four corners
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            # Filter out small regions
            if w > 1000 and h > 1000:  # Might need to adjust size threshold
                # Draw a rectangle around the detected boxes
                cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 0, 255), 30)
                # Fill the mask with a unique label for each contour
                cv2.drawContours(mask, [approx], -1, (label), thickness=cv2.FILLED)
                label += 1
                
    # Send a warning if an unexpected number of labels were identified
    total_labels = label-1
    if total_labels != 24:
        print(f"Warning: {total_labels} labels were identified when 24 labels were expected.")
        
    return mask


def ocr_read_labels(scan, mask):
    # Find unique labels
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    
    # Create a dictionary with keys from unique_labels
    label_dict = {int(label): None for label in unique_labels}
    
    # Loop through each label and extract the corresponding region
    for label in unique_labels:
        # Create a binary mask for the current label
        label_mask = np.zeros_like(mask, dtype=np.uint8)
        label_mask[mask == label] = 1
        
        # Crop the label from the scan
        crop_label = scan[np.ix_(label_mask.any(1), label_mask.any(0))]
    
        # Rotate label to be readable
        # crop_label = rotate(crop_label, 90)
    
        # Crop the number from the label
        crop_num = crop_label[400:800, 300:750] # TODO: Might need adjustment
    
        # _, crop_num = cv2.threshold(crop_num, 200, 255, cv2.THRESH_BINARY)
            
        # Reduce crop resolution for better ocr performence
        scale_percent = 0.50 
        new_width = int(crop_num.shape[1] * scale_percent)
        new_height = int(crop_num.shape[0] * scale_percent)
        crop_num = cv2.resize(crop_num, (new_width, new_height))
        
        # Identify the text
        ocr_result = pytesseract.image_to_string(crop_num, config=custom_config)
        
        label_dict[int(label)] = ocr_result.strip()

        # Show the region corresponding to the current label
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(7, 2))
        ax0.imshow(cv2.cvtColor(crop_label, cv2.COLOR_BGR2RGB))
        ax0.axis('off')
        ax1.imshow(cv2.cvtColor(crop_num, cv2.COLOR_BGR2RGB))
        ax1.axis('off')
        plt.show()
        print(ocr_result.strip())
        
    return label_dict

def process_and_save_wing(crop, image_outside_colored, contour, label, wing_position, file_output_dir, file_base_name, os=20):
    x, y, w, h = cv2.boundingRect(contour)

    wing_1 = crop[y-os:y+h+os, x-os:x+w+os]
    wing_1 = cv2.cvtColor(wing_1, cv2.COLOR_BGR2RGB)
    wing_1 = Image.fromarray(wing_1)
    wing_1.save(f"{file_output_dir}{file_base_name}_Label_{label}_{wing_position}_1.jpg")

    wing_2 = image_outside_colored[y-os:y+h+os, x-os:x+w+os]
    wing_2 = cv2.cvtColor(wing_2, cv2.COLOR_BGR2RGB)
    wing_2 = Image.fromarray(wing_2)
    wing_2.save(f"{file_output_dir}{file_base_name}_Label_{label}_{wing_position}_2.jpg")


def identify_wings(mask, scan, file_output_dir, file_base_name):
    # Find unique labels except for the background
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    
    # Loop through each label and extract the corresponding region
    for label in unique_labels:
        # Create a binary mask for the current label
        label_mask = np.zeros_like(mask, dtype=np.uint8)
        label_mask[mask == label] = 1

        # Extract the label from the scan
        crop = scan[np.ix_(label_mask.any(1), label_mask.any(0))]

        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # blurred_crop = cv2.medianBlur(gray_crop, 5)
        _, thresh = cv2.threshold(gray_crop, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50000]
    
        # Sort the contours from left to right (by the x-coordinate of their bounding box)
        sorted_contours = sorted(large_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
    
        # Draw contours on the image for visualization
        contour_image = crop.copy()
        cv2.drawContours(contour_image, large_contours, -1, (0, 0, 0), thickness=cv2.FILLED)
    
        # Fill the contour area on the mask
        wing_mask = np.zeros_like(contour_image[:, :, 0]) 
        wing_mask = cv2.drawContours(wing_mask, large_contours, -1, 255, thickness=cv2.FILLED)
        mask_inv = cv2.bitwise_not(wing_mask)

        # Convert the color to an array with the same shape as the image
        color_layer = np.full_like(contour_image, (255, 255, 255))
        
        image_outside_colored = np.where(mask_inv[:, :, None] == 255, color_layer, crop)
        # image_outside_colored = np.where(mask_inv, gray_crop)

        # Threshold between image and crop border
        # os = 20
        
        # Crop and save identified wings 
        if len(large_contours) == 2:
            process_and_save_wing(crop, image_outside_colored, sorted_contours[0], label, 'Left', file_output_dir, file_base_name)
            process_and_save_wing(crop, image_outside_colored, sorted_contours[1], label, 'Right', file_output_dir, file_base_name)
        
        elif len(large_contours) == 1:
            process_and_save_wing(crop, image_outside_colored, sorted_contours[0], label, 'Only', file_output_dir, file_base_name)
        
        else:
            print(f"Warning: {len(large_contours)} wings were identified on label {label}")


def main():
    # Ensure directories exist or create output directory
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' was not found.")
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all scans
    for filename in os.listdir(input_dir):
        input_file = os.path.join(input_dir, filename)
        file_base_name = filename.split(".")[0]
        file_output_dir = output_dir + file_base_name + "/"
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Check if it is a tif file
        if os.path.isfile(input_file) and filename.endswith('.tif'):
            print(f"Processing file: {filename}")
            scan = cv2.imread(input_file)
            scan = rotate(scan, 90)
            mask = generate_label_mask(scan)
            # label_dict = ocr_read_labels(scan, mask)
            identify_wings(mask, scan, file_output_dir, file_base_name)


if __name__=="__main__":
    main()
