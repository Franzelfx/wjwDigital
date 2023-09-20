import cv2
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt

def enhance_image(image_path, output_path):
    # Set a higher limit for the max image pixels attribute to avoid DecompressionBombError
    Image.MAX_IMAGE_PIXELS = None

    # Open the image using PIL
    image = Image.open(image_path).convert("RGB")  # Convert image to RGB mode

    # Convert the PIL Image to a NumPy array (required for OpenCV operations)
    image_np = np.array(image)

    # Upscale the image using cubic interpolation
    height, width = image_np.shape[:2]
    upscale_factor = 2  # Adjust the upscale factor as needed
    new_height, new_width = height * upscale_factor, width * upscale_factor
    image_np = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Define a kernel to enhance sharpness
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])

    # Apply the kernel to the image to enhance sharpness
    image_np = cv2.filter2D(image_np, -1, kernel)

    # Denoise the image using a bilateral filter
    # image_np = cv2.bilateralFilter(image_np, 9, 75, 75)

    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Further enhance the image using a CLAHE filter
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_np = clahe.apply(image_gray)

    # Save the enhanced image
    output_image = Image.fromarray(image_np)
    output_image.save(output_path)

def deskew(image):
    # Convert to grayscale if the image is in color
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Threshold the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find the coordinates of all non-zero pixels
    coords = np.column_stack(np.where(thresh > 0))

    # Find the rotated bounding box
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust the angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # Rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Print the angle of rotation
    print(f"[INFO] Angle of rotation: {angle:.3f}")

    return rotated




if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Enhance an image using adaptive thresholding and morphological operations.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("output_path", help="Path to save the enhanced image.")
    
    args = parser.parse_args()
    
    # Enhance the image
    enhance_image(args.image_path, args.output_path)
