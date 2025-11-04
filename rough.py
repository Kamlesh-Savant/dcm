import pydicom
import pytesseract
from PIL import Image
import numpy as np
import cv2
import os

# --- Configuration ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
file_path = '3.dcm'

try:
    # --- Step 1: Read DICOM image ---
    ds = pydicom.dcmread(file_path)
    pixel_array = ds.pixel_array

    # Normalize pixel values (0–255)
    image_2d = pixel_array.astype(float)
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
    image_2d_scaled = np.uint8(image_2d_scaled)

    # --- Step 2: Convert grayscale to BGR for Tesseract ---
    if len(image_2d_scaled.shape) == 2:
        final_image = cv2.cvtColor(image_2d_scaled, cv2.COLOR_GRAY2BGR)
    else:
        final_image = cv2.cvtColor(image_2d_scaled, cv2.COLOR_RGB2BGR)

    # --- Step 3: Enhance image for better OCR ---
    enhanced = cv2.equalizeHist(cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY))
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Upscale for better OCR accuracy
    scale_factor = 3
    upscaled = cv2.resize(thresh, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Save this image for debugging if needed
    cv2.imwrite("dicom_for_ocr.png", upscaled)
    print("✅ Saved enhanced image as dicom_for_ocr.png")

    # --- Step 4: OCR directly on the processed image ---
    print("Extracting text with Tesseract OCR...")
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.:/-()%"
    config_str = f'--psm 6 -c tessedit_char_whitelist="{whitelist}"'

    text = pytesseract.image_to_string(Image.fromarray(upscaled), config=config_str)

    # --- Step 5: Output ---
    print("\n--- Extracted Text from DICOM ---")
    if text.strip():
        print(text)
    else:
        print("❌ No text detected. Try adjusting brightness or contrast.")

except Exception as e:
    print(f"An error occurred: {e}")
