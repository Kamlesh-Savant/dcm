import cv2
import numpy as np
import pytesseract
from PIL import Image

# --- CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
image_path = "7.png"

# --- TUNING PARAMETER ---
# This is your "zoom" level.
# 2 means 2x larger, 3 means 3x larger. Start with 2 or 3.
scale_factor = 6

# --- Load the image with OpenCV ---
img_cv = cv2.imread(image_path)

# --- ======================================================= ---
# --- New: ZOOM (UPSCALE) THE IMAGE ---
# --- ======================================================= ---
# Get the original dimensions
width = int(img_cv.shape[1] * scale_factor)
height = int(img_cv.shape[0] * scale_factor)
dim = (width, height)

# Resize the image. cv2.INTER_CUBIC is a high-quality method for enlarging.
img_cv = cv2.resize(img_cv, dim, interpolation=cv2.INTER_CUBIC)
# --- ======================================================= ---


# --- PATH A: Process Green and Yellow Text (using HSV) ---
hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
combined_color_mask = cv2.bitwise_or(mask_green, mask_yellow)

# --- PATH B: Process White Text (using Grayscale Threshold) ---
# THIS IS THE CORRECTED LINE:
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

# Adjust the threshold value if needed.
_, mask_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# --- Combine all masks together ---
final_mask = cv2.bitwise_or(combined_color_mask, mask_white)

# --- Optional but Recommended: Refine the Mask ---
# This helps make characters more solid after upscaling.
kernel = np.ones((2,2), np.uint8)
final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

# --- Create a clean image for OCR ---
# Create a new white background image
final_image = np.full(img_cv.shape, 255, dtype=np.uint8)
# Use the final combined mask to copy the text (now in black) onto the white background
np.copyto(final_image, 0, where=final_mask[:,:,None] == 255)

# --- Perform OCR ---
final_image_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

# You can save the intermediate image to see what Tesseract is processing
# This is VERY helpful for debugging!
final_image_pil.save("processed_for_ocr.png")

# Optional but Recommended: Add Tesseract Configuration
# --psm 6 assumes a single uniform block of text.
custom_config = r'--oem 3 --psm 6'

text = pytesseract.image_to_string(final_image_pil, config=custom_config)

# --- Print the result ---
print(f"--- Extracted Text (Zoom Factor: {scale_factor}x) ---")
print(text.replace("\x0c", "").replace("_","").replace(" + ","").replace(";","").replace(" m ","").replace("fr","").strip())