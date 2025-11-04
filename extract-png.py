from PIL import Image
import pytesseract

# Path to tesseract.exe (update if different on your computer)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open the image
img = Image.open("out.png")

# Convert to grayscale (makes it easier for OCR)
img = img.convert("L")

# Extract text from the image
text = pytesseract.image_to_string(img)

# Remove extra characters and print the text
print(text.replace("\x0c", "").strip())