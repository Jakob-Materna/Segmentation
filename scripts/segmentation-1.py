import os
import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate
import pytesseract
import sys
from skimage.segmentation import clear_border

# Path to pytesseract-ocr executable. See readme for more information.
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
custom_config='--psm 10 --oem 3 -c tessedit_char_whitelist=wW0123456789'
# custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
# custom_config = '--psm 6'


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


def generate_border(mask, image, border_size):
    # Erode the mask to shrink it slightly
    erosion_kernel = np.ones((3, 3), np.uint8)      # Kernel for erosion
    eroded_mask = cv2.erode(mask, erosion_kernel, iterations=1)  
 
    # Define the kernel size for dilation based on the desired border size (Add 1 to keep it odd)
    kernel_size = 2 * border_size + 1 
    dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)   # Kernel for dilation
    dilated_mask = cv2.dilate(eroded_mask, dilation_kernel, iterations=1)

    # Replace 255 values to 127 for all pixels in the dilated mask (border pixels)
    dilated_127_mask = np.where(dilated_mask == 255, 127, dilated_mask) 
    
    # In the dilated mask, convert the eroded object parts to pixel value 255
    # What's remaining with a value of 127 would be the boundary pixels.
    mask_with_border = np.where(eroded_mask > 127, 255, dilated_127_mask)

    # Now, apply this border mask to the original image to draw the border
    # First, create a copy of the original image
    image_with_border = image.copy()
    
    # Convert the mask to a binary form where 127 represents border pixels
    border_pixels = mask_with_border == 127
    
    # Apply the border to the original image where border_pixels is True
    image_with_border[border_pixels] = (0, 0, 0)
    
    return image_with_border


def remove_noise(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    cleaned_image = np.zeros_like(image)
    cv2.drawContours(cleaned_image, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return cleaned_image


def identify_segments(mask, scan, output_base_name, bin_threshold=110):
    # Find unique labels
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    # unique_labels = unique_labels[unique_labels == 1]
    
    # Loop through each label and extract the corresponding region
    for label in unique_labels:
        # Create a binary mask for the current label
        label_mask = np.zeros_like(mask, dtype=np.uint8)
        label_mask[mask == label] = 1
        
        # Extract the label from the scan
        crop = scan[np.ix_(label_mask.any(1), label_mask.any(0))]
        # crop = rotate(crop, 90)
    
        
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
        # cv2.drawContours(wing_contour_image, large_contours, -1, (0, 0, 0), 5)
        cv2.drawContours(contour_image, large_contours, -1, (0, 0, 0), thickness=cv2.FILLED)
    
        mask = np.zeros_like(contour_image[:, :, 0]) 
        # Fill the contour area on the mask
        mask = cv2.drawContours(mask, large_contours, -1, 255, thickness=cv2.FILLED)
        mask_inv = cv2.bitwise_not(mask)
    
        # Define the color you want to fill the outside with (e.g., red)
        fill_color = (255, 255, 255)
        
        # Convert the color to an array with the same shape as the image
        color_layer = np.full_like(contour_image, fill_color)
        
        image_outside_colored = np.where(mask_inv[:, :, None] == 255, color_layer, crop)
        # image_outside_colored = np.where(mask_inv, gray_crop)
        
        # plt.figure(figsize=(5, 5))
        # plt.imshow(cv2.cvtColor(image_outside_colored, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()
        
        bordered_wings = generate_border(mask_inv, image_outside_colored, 5)
        
        # plt.figure(figsize=(5, 5))
        # plt.imshow(cv2.cvtColor(bordered_wings, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()
    
        
        gray_bordered_wings = cv2.cvtColor(bordered_wings, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_bordered_wings, bin_threshold, 255, cv2.THRESH_BINARY)
    
        # plt.figure(figsize=(5, 5))
        # plt.imshow(cv2.cvtColor(binary_image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()
    
        # binary_image = remove_noise(binary_image)
        # kernel = np.ones((5,5), np.uint8)  # You can adjust the kernel size
        # binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    
        # Threshold between image and crop border
        os = 20
    
        # Crop identified wings 
        if (len(large_contours) == 2):
            x, y, w, h = cv2.boundingRect(sorted_contours[0])
            left_wing = binary_image[y-os:y+h+os, x-os:x+w+os]
            left_wing_rgb = crop[y-os:y+h+os, x-os:x+w+os]
            
            x, y, w, h = cv2.boundingRect(sorted_contours[1])
            right_wing = binary_image[y-os:y+h+os, x-os:x+w+os]
            right_wing_rgb = crop[y-os:y+h+os, x-os:x+w+os]
    
            # img_num = str(bin_threshold)
            # left_wing = Image.fromarray(left_wing) 
            # left_wing.save(f"../../images/binary_left_{img_num}.jpg")
            # right_wing = Image.fromarray(right_wing)
            # right_wing.save(f"../../images/binary_right_{img_num}.jpg")
            
            # print(f"Label: {label}")
            # print(f"Number of contours: {len(large_contours)}")
            # print("\n\n\n\n\n\n")
    
            # Find contours
            left_wing = clear_border(left_wing)
            # left_wing = cv2.bitwise_not(left_wing)
            contours, _ = cv2.findContours(left_wing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
            contours = large_contours
            
            # Sort the contours from left to right (by the x-coordinate of their bounding box)
            # sorted_contours = sorted(large_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
        
            # Draw contours on the image for visualization
            left_wing_contour_image = left_wing.copy()
            left_wing_contour_image = cv2.cvtColor(left_wing_contour_image, cv2.COLOR_GRAY2RGB)
            # cv2.drawContours(left_wing_contour_image, large_contours, -1, (0, 0, 255), 5)
        
    
            # Calculate and display the size of each contour (segment)
            segment_areas = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                segment_areas.append(area)
                
                # Draw each contour for visualization
                cv2.drawContours(left_wing_rgb, [contour], -1, (0, 0, 255), 2)
                # Display area as text on the image
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])  # Centroid x
                    cy = int(M['m01'] / M['m00'])  # Centroid y
                    cv2.putText(left_wing_rgb, f'{int(area)}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Display the wing with contours and segment areas
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(left_wing_rgb, cv2.COLOR_BGR2RGB))
            plt.axis('on')
            plt.savefig(output_base_name + "_" + str(label) + "_lw.jpg")
            # plt.show()
            
            # Print the areas of each segment
            # print("Areas of the wing segments:")
            # for i, area in enumerate(segment_areas):
            #     print(f"Segment {i + 1}: {area} pixels")

            # Find contours
            right_wing = clear_border(right_wing)
            # right_wing = cv2.bitwise_not(right_wing)
            contours, _ = cv2.findContours(right_wing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
            contours = large_contours
            
            # Sort the contours from left to right (by the x-coordinate of their bounding box)
            # sorted_contours = sorted(large_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
        
            # Draw contours on the image for visualization
            right_wing_contour_image = right_wing.copy()
            right_wing_contour_image = cv2.cvtColor(right_wing_contour_image, cv2.COLOR_GRAY2RGB)
            # cv2.drawContours(left_wing_contour_image, large_contours, -1, (0, 0, 255), 5)
        
    
            # Calculate and display the size of each contour (segment)
            segment_areas = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                segment_areas.append(area)
                
                # Draw each contour for visualization
                cv2.drawContours(right_wing_rgb, [contour], -1, (0, 0, 255), 2)
                # Display area as text on the image
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])  # Centroid x
                    cy = int(M['m01'] / M['m00'])  # Centroid y
                    cv2.putText(right_wing_rgb, f'{int(area)}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Display the wing with contours and segment areas
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(right_wing_rgb, cv2.COLOR_BGR2RGB))
            plt.axis('on')
            plt.savefig(output_base_name + "_" + str(label) + "_rw.jpg")
            # plt.show()
            
            # Print the areas of each segment
            # print("Areas of the wing segments:")
            # for i, area in enumerate(segment_areas):
            #     print(f"Segment {i + 1}: {area} pixels")


def main():
    # Define directories
    input_dir = "/mnt/c/Projects/Master/Testdata/Scans/"
    output_dir = "/mnt/c/Projects/Master/Testdata/Temp/"

    # Ensure directories exist or create output directory
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' was not found.")

    os.makedirs(output_dir, exist_ok=True)

    # Loop through all scans
    for filename in os.listdir(input_dir):
        input_file = os.path.join(input_dir, filename)  
        
        # Check if it is a tif file
        if os.path.isfile(input_file) and filename.endswith('.tif'):
            print(f"Processing file: {filename}")
            scan = cv2.imread(input_file)
            scan = rotate(scan, 90)
            mask = generate_label_mask(scan)
            # label_dict = ocr_read_labels(scan, mask)
            output_base_name = output_dir + filename.split(".")[0]
            identify_segments(mask, scan, output_base_name)


if __name__=="__main__":
    main()