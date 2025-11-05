import easyocr
import cv2
import time

def extract_text_with_easyocr(image_path, show_image=False):
    """
    Extracts text from an image using EasyOCR, with optional preprocessing.
    
    Args:
        image_path (str): The path to the image file.
        show_image (bool): If True, displays the image with detected text boxes.

    Returns:
        list: A list of extracted text strings.
    """
    print("Initializing EasyOCR reader...")
    # Initialize the EasyOCR reader for English.
    # Set gpu=False if you don't have a compatible GPU or CUDA setup.
    # It will run on the CPU.
    reader = easyocr.Reader(['en'], gpu=False) 
    print("Reader initialized.")

    try:
        # 1. Load the image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return ["Error: Image not found or could not be read."]

        # --- OPTIONAL BUT RECOMMENDED PREPROCESSING ---
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to create a binary image. This helps isolate the text.
        # OTSU's method is excellent for finding an optimal threshold value automatically.
        _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # For ultrasound's white-on-dark text, an inverted threshold can work better.
        # _, binary_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) # Alternative
        #----------------------------------------------

        print(f"Reading text from '{image_path}'...")
        start_time = time.time()
        
        # 2. Use EasyOCR to read text from the preprocessed image
        # We pass the image array directly to readtext
        results = reader.readtext(binary_img)
        
        end_time = time.time()
        print(f"Text extraction took {end_time - start_time:.2f} seconds.")

        extracted_text = []
        # The result is a list of tuples: (bounding_box, text, confidence_score)
        for (bbox, text, prob) in results:
            # You can filter by confidence score if needed
            if prob > 0.4: # Example threshold: only include text with >40% confidence
                print(f'Detected text: "{text}" (Confidence: {prob:.2f})')
                extracted_text.append(text)
                
                # If show_image is True, draw the bounding box on the original image
                if show_image:
                    # Get the top-left and bottom-right coordinates
                    (tl, tr, br, bl) = bbox
                    tl = (int(tl[0]), int(tl[1]))
                    br = (int(br[0]), int(br[1]))
                    
                    # Draw a rectangle and put the detected text
                    cv2.rectangle(img, tl, br, (0, 255, 0), 2) # Green box
                    cv2.putText(img, text, (tl[0], tl[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3. Display the image with bounding boxes if requested
        if show_image:
            # Resize for display if the image is too large
            scale_percent = 50 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

            cv2.imshow("Detection Result", resized)
            cv2.waitKey(0) # Wait for a key press to close the image window
            cv2.destroyAllWindows()
            
        return extracted_text

    except Exception as e:
        return [f"An error occurred: {e}"]

# --- USAGE EXAMPLE ---
image_file = '7.png'

# Set 'show_image' to True to see a visual of what EasyOCR detected
all_text = extract_text_with_easyocr(image_file, show_image=True)

print("\n--- Summary of Extracted Text ---")
if all_text:
    for line in all_text:
        print(line)
else:
    print("No text was extracted.")
