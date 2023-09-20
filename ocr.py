import re
import os
import sys
import cv2
import PIL
import logging
import argparse
import numpy as np
import pytesseract
from PIL import Image
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

PATTERNS = [
    r"\d{2}-\w{10}",
    r"\d{2}-\d+-\d{2}-\d"
]
PIL.Image.MAX_IMAGE_PIXELS = None

# Log config to file
logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def log_and_print(message, level=logging.INFO, file_only=False, qt_text_edit=None):
    log_entry = f"{logging.getLevelName(level)} - {message}"
    logging.log(level, log_entry)
    if level >= logging.ERROR and not file_only:
        print(message, file=sys.stderr)
    elif not file_only:
        print(message)
    if qt_text_edit:
        qt_text_edit.append(log_entry)

class OCRScan:
    def __init__(self, tesseract_path="tesseract", overlap_percentage=30, section_size_percentage=70, whitelist="0123456789A-"):
        log_and_print("Initializing OCRScan", level=logging.DEBUG)
        self.tesseract_path = tesseract_path
        self.overlap_percentage = overlap_percentage
        self.section_size_percentage = section_size_percentage
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.whitelist = whitelist

    def _preprocess_image(self, image_path_or_obj, enhance=False):
        # If it's a string path, open it, otherwise assume it's an Image object
        if isinstance(image_path_or_obj, str):
            image = Image.open(image_path_or_obj)
        else:
            image = image_path_or_obj
            # Convert to grayscale        
        # Apply image enhancement if needed
        if enhance:
            # Open the image using PIL
            image.convert("RGB")  # Convert image to RGB mode

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

            # Convert the image to grayscale
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

            # Further enhance the image using a CLAHE filter
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_np = clahe.apply(image_gray)

            # Save the enhanced image
            image = Image.fromarray(image_np)
        else:
            image = image.convert("L")
        
        return image

    def _ocr_image(self, image_obj=None):
        try:
            custom_config = f"-c tessedit_char_whitelist={self.whitelist} --oem 3 --psm 6"
            image_obj = self._preprocess_image(image_obj)
            text = pytesseract.image_to_string(image_obj, config=custom_config)
            return text
        except Exception as e:
            log_and_print(f"An error occurred: {e}", level=logging.ERROR)
            return None

    def _ocr_on_section(
        self,
        section,
        x,
        y,
        ocr_instance,
        patterns,
        output_folder,
        md_file,
    ):
        try:
            output_filename = f"section_{x}_{y}"
            output_path = os.path.join(output_folder, f"{output_filename}.png")

            #section.save(output_path)
            section_text = ocr_instance._ocr_image(image_obj=section)

            # Write to txt file
            txt_file_path = os.path.join(output_folder, f"{output_filename}.txt")
            with open(txt_file_path, "w") as f:
                f.write(section_text)

            # Postprocess the text
            log_and_print(
                f"Performing postprocessing on section {output_path}",
                level=logging.DEBUG,
            )
            section_text = ocr_instance._postprocess(section_text, patterns)

            if section_text:
                # Add section image and text to the markdown file
                md_file.write(f"![Section Image](./sections/{output_filename})\n\n")
                md_file.write(f"```\n{section_text}\n```\n\n")
            return section_text
        except Exception as e:
            log_and_print(f"An error occurred in worker: {e}", level=logging.ERROR)
            return None
        
    def _find_most_common_result(self, results):
        """
        Find the most common result from a list of results.

        Args:
            results (list): A list of results (strings).

        Returns:
            str or None: The most common result or None if the list is empty.
        """
        if not results:
            return None

        # Filter None and empty strings from results
        results = [res for res in results if res]

        # If we have only one result, return it
        if len(results) == 1:
            return results[0]

        # Create a Counter object to count occurrences of each result
        result_counter = Counter(results)

        # Find the most common result (the one with the most matches)
        most_common_result = result_counter.most_common(1)

        return most_common_result[0][0] if most_common_result else None


    def _postprocess(self, text, patterns):
        try:
            matched_text = []
            for pattern in patterns:
                matches = re.findall(pattern, text)
                matched_text.extend(matches)
            filtered_text = matched_text[0] if matched_text else None
            # Output Format should be:
            # xx-xxxxxx-xx-xx (if length is 13)
            if filtered_text and len(filtered_text) == 13:
                filtered_text = filtered_text.replace("-", "")
                filtered_text = filtered_text[:2] + "-" + filtered_text[2:8] + "-" + filtered_text[8:10] + "-" + filtered_text[10:]
            return filtered_text
        except Exception as e:
            log_and_print(f"An error occurred: {e}", level=logging.ERROR)
            return None
        
    def ocr_on_image(
        self,
        image_path,
        patterns=PATTERNS,
        run_again_with_enhanced_image=True,
    ):
        try:
            log_and_print(
                f"Performing OCR with sliding window on image {image_path}",
                level=logging.DEBUG,
            )
            # Create a dynamic output folder based on the input file name in the same directory but create "results" folder
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            output_folder = os.path.join(
                os.path.dirname(image_path), "results", base_filename
            )
            # Create a log file to store the OCR results
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Create a markdown file to store the OCR results
            md_file_path = os.path.join(output_folder, "OCR_Results.md")
            md_file = open(md_file_path, "w")
            image = Image.open(image_path)
            image_width, image_height = image.size
            section_width = int(image_width * self.section_size_percentage / 100)
            section_height = int(image_height * self.section_size_percentage / 100)
            shift_width = section_width - int(image_width * self.overlap_percentage / 100)
            shift_height = section_height - int(image_height * self.overlap_percentage / 100)

            results = []

            with ThreadPoolExecutor() as executor:
                future_results = []

                for y in range(0, image_height, shift_height):
                    for x in range(0, image_width, shift_width):
                        right = min(x + section_width, image_width)
                        bottom = min(y + section_height, image_height)
                        section = image.crop((x, y, right, bottom))

                        future = executor.submit(
                        self._ocr_on_section,
                        section,
                        x,
                        y,
                        self,
                        patterns,
                        output_folder,
                        md_file,
                    )
                        future_results.append(future)
            
                results = [future.result() for future in future_results]
            
            # Close the Markdown file
            md_file.close()
            # Return the first non-empty or non-None result
            # If result is none, run again with enhanced image
            if not self._find_most_common_result(results) and run_again_with_enhanced_image:
                log_and_print(
                    f"No results found, running again with enhanced image",
                    level=logging.INFO,
                )
                return self.ocr_on_image(
                    image_path,
                    patterns,
                    run_again_with_enhanced_image=False,
                )
            else:
                return self._find_most_common_result(results)
        except Exception as e:
            log_and_print(f"An error occurred: {e}", level=logging.ERROR)
            return None

def main():
    parser = argparse.ArgumentParser(description="Perform OCR on an image.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("--tp", default="tesseract", help="Path to Tesseract executable.")
    parser.add_argument("--op", type=int, default=0, help="Overlap percentage for sections.")
    parser.add_argument("--ssp", type=int, default=100, help="Section size percentage.")
    parser.add_argument("--lf", default="debug.log", help="Path to log file.")
    parser.add_argument("--whitelist", default="0123456789A-", help="Whitelist for Tesseract.")
    
    args = parser.parse_args()
    
    ocr_scan = OCRScan(
        tesseract_path=args.tp,
        overlap_percentage=args.op,
        section_size_percentage=args.ssp,
        whitelist=args.whitelist,
    )
    print(ocr_scan.ocr_on_image(args.image_path))

if __name__ == "__main__":
    main()
