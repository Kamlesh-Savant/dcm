import pydicom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_windowing(image_array, window_center, window_width):
    """
    Applies windowing to a NumPy array representing an image.
    This adjusts the brightness and contrast to a viewable range.
    """
    # Formula to get the min and max values for the window
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    
    # Clip the array to the window
    clipped_array = np.clip(image_array, window_min, window_max)
    
    # Scale the clipped array to 0-255 (for an 8-bit image)
    scaled_array = ((clipped_array - window_min) / (window_width)) * 255.0
    
    # Convert to unsigned 8-bit integer
    return scaled_array.astype(np.uint8)

def process_dicom_with_windowing(dicom_path, output_path):
    """
    Reads a DICOM file, applies windowing, and saves as a PNG.
    """
    try:
        # Read the DICOM file
        ds = pydicom.dcmread(dicom_path)

        # Get the raw pixel data
        pixel_array = ds.pixel_array

        # Get windowing values from DICOM tags (if they exist)
        # (0028, 1050) = Window Center, (0028, 1051) = Window Width
        window_center = ds.WindowCenter
        window_width = ds.WindowWidth
        
        # If WC/WW are multi-valued, just use the first one
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = window_center[0]
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = window_width[0]

        # Apply windowing
        windowed_array = apply_windowing(pixel_array, window_center, window_width)
        
        # Create and save the image
        image = Image.fromarray(windowed_array)
        image.save(output_path)
        print(f"Successfully saved windowed image to {output_path}")

        # --- Optional: Display the image using matplotlib ---
        plt.imshow(windowed_array, cmap='gray')
        plt.title("DICOM Image with Windowing")
        plt.axis('off') # Hide axes
        plt.show()

    except AttributeError:
        print(f"Warning: DICOM file {dicom_path} does not have WindowCenter/WindowWidth tags. Using simple normalization.")
        # Fallback to the simple method if windowing tags are not present
        save_dicom_as_png(dicom_path, output_path)
    except Exception as e:
        print(f"An error occurred: {e}")

def save_dicom_as_png(dicom_file_path, output_png_path):
    """
    Reads a DICOM file, extracts the pixel data, and saves it as a PNG image.

    Args:
        dicom_file_path (str): The path to the input .dcm file.
        output_png_path (str): The path where the output .png file will be saved.
    """
    try:
        # 1. Read the DICOM file
        dicom_dataset = pydicom.dcmread(dicom_file_path)

        # 2. Get the pixel data as a NumPy array
        pixel_array = dicom_dataset.pixel_array

        # 3. Normalize the pixel array to be in the 0-255 range for an 8-bit image
        # This is a simple normalization, for medical images, windowing is better (see below)
        pixel_array_normalized = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
        pixel_array_scaled = (pixel_array_normalized * 255).astype(np.uint8)

        # 4. Create an image from the array using Pillow
        image = Image.fromarray(pixel_array_scaled)

        # 5. Save the image as a PNG
        image.save(output_png_path)
        print(f"Successfully saved image to {output_png_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


# --- Usage Example ---
input_dcm_advanced = '3.dcm'
output_png_windowed = 'output_image_windowed.png'

process_dicom_with_windowing(input_dcm_advanced, output_png_windowed)
