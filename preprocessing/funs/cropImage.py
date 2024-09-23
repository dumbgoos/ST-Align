## test for match the sequence and image
import matplotlib.pyplot as plt
from PIL import Image

import scanpy as sc
import pandas as pd
import numpy as np

import os

# data

def scale_coordinate(data):
    """Convert imagecol and imagerow into high-resolution coordinates."""
    library_id = list(data.uns["spatial"].keys())[0]
    scale = data.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]
    if isinstance(data.obsm["spatial"][1, 1], str):
        data.obsm["spatial"] = data.obsm["spatial"].astype("float")
    image_coordinates = data.obsm["spatial"] * scale
    data.obs["imagecol"] = image_coordinates[:, 0]
    data.obs["imagerow"] = image_coordinates[:, 1]
    return data

def calculate_patch_size_in_pixels(patch_size_um, scalefactors, spot_size_um):
    """
    Calculate the number of pixels for a given patch size in micrometers.
    
    Parameters:
    patch_size_um (float): The desired patch size in micrometers.
    scalefactors (dict): The scalefactors output by space ranger
    spot_size_um (float): The diameter of the spot in micrometers.

    Returns:
    int: The patch size in pixels.
    """
    # Calculate the 1um in hires image
    tissue_hires_scalef = scalefactors["tissue_hires_scalef"]
    spot_diameter_fullres = scalefactors["spot_diameter_fullres"]

    hires_1um_per_pixel = tissue_hires_scalef * spot_diameter_fullres / spot_size_um

    return int(hires_1um_per_pixel * patch_size_um)

def get_spot_diameter_pixels(data, library_id):
    """Get the diameter of a spot in pixels in the high-resolution image."""
    scalefactors = data.uns["spatial"][library_id]["scalefactors"]
    spot_diameter_fullres = scalefactors["spot_diameter_fullres"]
    tissue_hires_scalef = scalefactors["tissue_hires_scalef"]
    return spot_diameter_fullres * tissue_hires_scalef

def normalize_image(image, mean, std):
    """
    Normalize a single image.
    
    Parameters:
    - image: A numpy array of shape (width, height, number_of_channels)
    - mean: List of mean values for each channel
    - std: List of standard deviation values for each channel
    
    Returns:
    - normalized_image: A numpy array of normalized image
    """
    # Convert mean and std to arrays and reshape for broadcasting
    mean = np.array(mean).reshape(1, 1, -1)
    std = np.array(std).reshape(1, 1, -1)
    
    # Normalize image
    normalized_image = (image - mean) / std
    
    return normalized_image

def crop_images_by_physical_size(data, patch_size_um):
    """Crop image based on patch size in micrometers and return an array of cropped images."""
    # Step 1: Scale coordinates
    data = scale_coordinate(data)
    
    # Step 2: Get necessary scale factors and image data
    library_id = list(data.uns["spatial"].keys())[0]
    scalefactors = data.uns["spatial"][library_id]["scalefactors"]
    spot_size_um = 55  # Assuming the diameter of the spot in micrometers
    
    # Step 3: Calculate patch size in pixels
    patch_size_pixels = calculate_patch_size_in_pixels(patch_size_um, scalefactors, spot_size_um)
    
    # Step 4: Get high-resolution image data
    raw_image_data = data.uns["spatial"][library_id]["images"]["hires"]
    image_data = normalize_image(raw_image_data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Step 5: Crop images
    cropped_images = []
    expression_values = []
    imagecol = []
    imagerow = []

    for row, col in zip(data.obs["imagerow"], data.obs["imagecol"]):
        left = int(col - patch_size_pixels // 2)
        top = int(row - patch_size_pixels // 2)
        right = int(col + patch_size_pixels // 2)
        bottom = int(row + patch_size_pixels // 2)
        
        # Ensure the cropped area is within the image boundaries
        if left < 0 or top < 0 or right > image_data.shape[1] or bottom > image_data.shape[0]:
            continue
        
        cropped_image = image_data[top:bottom,left:right]
        cropped_images.append(cropped_image)

        # Get the spot coordinates within the patch
        spots_in_patch = data[(data.obs["imagecol"] >= left) & (data.obs["imagecol"] <= right) &
                              (data.obs["imagerow"] >= top) & (data.obs["imagerow"] <= bottom)]
        
        if len(spots_in_patch) > 0:
            # Calculate average expression vector
            average_expression = spots_in_patch.X.toarray().mean(axis=0)
            expression_values.append(average_expression)
            imagecol.append(col)
            imagerow.append(row)
        
    # Step 6: stack sequence vector and cropped images
    cropped_images_array = np.stack(cropped_images) if cropped_images else np.array([])
    sequence_vector = np.stack(expression_values) if expression_values else np.array([])

    # Step 7: record the name of sequence and image
    row_vector = np.stack(imagerow) if imagerow else np.array([])    
    col_vector = np.stack(imagecol) if imagecol else np.array([])
    coor_dataframe = pd.DataFrame({
        'row': row_vector,
        'col': col_vector
    })

    return sequence_vector, cropped_images_array, coor_dataframe

def crop_images_by_pixel_size(data, patch_size_pixels):
    """Crop image based on patch size in pixels and return an array of cropped images and expression vectors."""
    # Step 1: Scale coordinates
    data = scale_coordinate(data)
    
    # Step 2: Get the spot diameter in pixels
    library_id = list(data.uns["spatial"].keys())[0]
    spot_diameter_pixels = get_spot_diameter_pixels(data, library_id)
    
    # Step 3: Get high-resolution image data
    image_data = data.uns["spatial"][library_id]["images"]["hires"]
    img = Image.fromarray((image_data * 255).astype(np.uint8))
    
    # Step 4: Crop images and calculate average expression
    cropped_images = []
    expression_values = []
    for row, col in zip(data.obs["imagerow"], data.obs["imagecol"]):
        left = int(col - patch_size_pixels // 2)
        top = int(row - patch_size_pixels // 2)
        right = int(col + patch_size_pixels // 2)
        bottom = int(row + patch_size_pixels // 2)
        
        # Ensure the cropped area is within the image boundaries
        if left < 0 or top < 0 or right > image_data.shape[1] or bottom > image_data.shape[0]:
            continue
        
        cropped_image = np.array(img.crop((left, top, right, bottom)))
        cropped_images.append(cropped_image)
        
        # Get the spot coordinates within the patch
        spots_in_patch = data[(data.obs["imagecol"] >= left) & (data.obs["imagecol"] <= right) &
                              (data.obs["imagerow"] >= top) & (data.obs["imagerow"] <= bottom)]
        
        if len(spots_in_patch) > 0:
            # Calculate average expression vector
            average_expression = spots_in_patch.X.toarray().mean(axis=0)
            expression_values.append(average_expression)
    
    cropped_images_array = np.stack(cropped_images) if cropped_images else np.array([])
    sequence_vector = np.stack(expression_values) if expression_values else np.array([])
    
    return sequence_vector, cropped_images_array