import time
from pathlib import Path
import os
import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov

#load the model
model_xml_path = 'MiDaS_small.xml'

core = ov.Core()
core.set_property({'CACHE_DIR': '../cache'})
model = core.read_model(model_xml_path)
compiled_model = core.compile_model(model=model)

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)

network_input_shape = list(input_key.shape)
network_image_height, network_image_width = network_input_shape[2:]
 
def normalize_minmax(data):
    """Normalizes the values in `data` between 0 and 1"""
    return (data - data.min()) / (data.max() - data.min())


def convert_result_to_image(result, colormap="viridis"):
    """
    Convert network result of floating point numbers to an RGB image with
    integer values from 0-255 by applying a colormap.

    `result` is expected to be a single network result in 1,H,W shape
    `colormap` is a matplotlib colormap.
    See https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    cmap = matplotlib.cm.get_cmap(colormap)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255
    result = result.astype(np.uint8)
    return result


def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

def process_single_image(file_path, save_folder_path):
    image = cv2.imread(file_path)  
    resized_image = cv2.resize(src=image, dsize=(network_image_height, network_image_width))
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
    result = compiled_model([input_image])[output_key]
    result_image = convert_result_to_image(result=result)
    result_image = cv2.resize(result_image, image.shape[:2][::-1])  
    grayscale_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    save_path = os.path.join(save_folder_path, os.path.splitext(os.path.basename(file_path))[0] + '_grayscale.jpg')

    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    cv2.imwrite(save_path, grayscale_image)

    print(f"Converted and saved grayscale image to {save_path}")
    
# Example usage
file_path = 'D:/CV+GNN/application/test_image/r01.png'  
save_folder_path = 'D:/CV+GNN/application/test_grey'    
process_single_image(file_path, save_folder_path)    