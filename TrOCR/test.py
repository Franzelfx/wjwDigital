import re
import os
import PIL
import sys
import shutil
import logging
import pytesseract
import pandas as pd
from io import StringIO
import PyQt5.QtCore as QtCore
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import QThreadPool, QRunnable, pyqtSignal

# Log config to file
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

PIL.Image.MAX_IMAGE_PIXELS = 933120000

INPUT_DIR = "./MFLK_Ohne Hollerith TEST UHDE/11_005000"
OUTPUT_DIR = f"{INPUT_DIR}/results"
OVERLAP_PERCENTAGE = 20
SECTION_SIZE_PERCENTAGE = 50
# Pattern can be xx-xxxxxx|dd|aa or xx-xxxxxx1dd1aa
#                xx-xxxxxx|-dd|-a or xx-xxxxxx1-dd1-a
#                xx-xxxxxx|-dd-d or xx-xxxxxx1-dd-d
# or combinations of the above
PATTERNS = [
    r"\d{2}-\d{6}\|\w{2}\|\w{2}",  # xx-xxxxxx|dd|aa
    r"\d{2}-\d{6}\|\-\w{2}\|\-\w",  # xx-xxxxxx|-dd|-a
    r"\d{2}-\d{6}1\w{2}1\w{2}",  # xx-xxxxxx1dd1aa
    r"\d{2}-\d{6}1\-\w{2}1\-\w",  # xx-xxxxxx1-dd1-a
]
# detect A,o,O 0-9, /\!|Ii()[]{}<>- and space
CHAR_WHITELIST = "0123456789AOolL!/\|Ii()[]{}<>- "
REPLACEMENTS = {
    "O": "0",
    "o": "0",
    "l": "|",
    "L": "|",
    "I": "|",
    "{": "|",
    "}": "|",
    "!": "|",
    "[": "|",
    "]": "|",
    "(": "|",
    ")": "|",
    "<": "|",
    ">": "|",
    "/": "|",
    "\\": "|",
}

def log_and_print(message, level=logging.INFO, file_only=False):
    logging.log(level, message)
    if level >= logging.ERROR and not file_only:
        print(message, file=sys.stderr)
    elif not file_only:
        print(message)

class OCRScan:
    def __init__(self):
        # Automatically detect the path to the Tesseract executable
        log_and_print("Initializing OCRScan", level=logging.DEBUG)
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

    def enhance_image(self, image):
        if image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.5)  # Increase brightness by 1.5 times

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)  # Increase contrast by 1.5 times

        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)  # Increase sharpness by 2 times
        
        return image

    def ocr_image(self, image_path=None, image_obj=None):
        try:
            # Create a dynamic output folder based on the input file name
            base_filename = os.path.basename(image_path).split('.')[0]
            # txt file path is parent of the image file (remove the image file name)
            txt_file_path = os.path.join(os.path.dirname(image_path), f"{base_filename}.txt")
            if image_path:
                image = Image.open(image_path)
            elif image_obj:
                image = image_obj
            else:
                raise ValueError("Either image_path or image_obj should be provided.")
            
            # Enhance the image
            image = self.enhance_image(image)

            # Perform OCR on the image with specified configurations
            text = pytesseract.image_to_string(
                image,
                config='--psm 6 --oem 3 -c tessedit_char_whitelist={}'.format(CHAR_WHITELIST)
            )

            # Write the OCR results to a text file
            with open(txt_file_path, 'w') as f:
                f.write(text)

            return text
        
        except Exception as e:
            log_and_print(f"An error occurred: {e}", level=logging.ERROR)
            return None
        
    def ocr_sliding_window(self, image_path, patterns=None, section_size_percentage=70, overlap_percentage=30):
        try:
            log_and_print(f"Performing OCR with sliding window on image {image_path}", level=logging.DEBUG)
            
            # Create a dynamic output folder based on the input file name
            base_filename = os.path.basename(image_path).split('.')[0]
            output_folder = os.path.join(OUTPUT_DIR, base_filename, "sections")
            # Create a log file to store the OCR results
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # Create a markdown file to store the OCR results
            md_file_path = os.path.join(OUTPUT_DIR, base_filename, "OCR_Results.md")
            md_file = open(md_file_path, 'w')
            
            image = Image.open(image_path)
            image_width, image_height = image.size
            section_width = int(image_width * section_size_percentage / 100)
            section_height = int(image_height * section_size_percentage / 100)
            shift_width = section_width - int(image_width * overlap_percentage / 100)
            shift_height = section_height - int(image_height * overlap_percentage / 100)
            
            results = []
            file_data = []  # List to store file data for CSV

            for y in range(0, image_height, shift_height):
                for x in range(0, image_width, shift_width):
                    right = min(x + section_width, image_width)
                    bottom = min(y + section_height, image_height)
                    section = image.crop((x, y, right, bottom))
                    
                    output_filename = f"section_{x}_{y}.png"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    section.save(output_path)
                    log_and_print(f"Performing OCR on section {output_path}", level=logging.DEBUG)
                    section_text = self.ocr_image(image_obj=section, image_path=output_path)
                    # Postprocess the text
                    section_text = self._postprocess(section_text, patterns)
                    
                    if section_text:
                        results.append(section_text)
                        
                        # Add section image and text to the markdown file
                        md_file.write(f"![Section Image](./sections/{output_filename})\n\n")
                        md_file.write(f"```\n{section_text}\n```\n\n")

                        file_data.append({
                            "Input File": image_path,
                            "Output File": output_filename,
                            "Extracted Text": section_text
                        })
            
            # Close the markdown file
            md_file.close()

            # Create a pandas DataFrame from the file_data list
            file_df = pd.DataFrame(file_data)
            
            # Save the DataFrame to a CSV file
            csv_filename = "extracted_data.csv"
            csv_path = os.path.join(output_folder, csv_filename)
            file_df.to_csv(csv_path, index=False)
            
            return "\n".join(results)
        except Exception as e:
            log_and_print(f"An error occurred: {e}", level=logging.ERROR)
            return None

    def replace_10th_and_13th_char_if_no_dash(self, s):
        # Remove the initial dash from consideration
        s_without_initial_dash = s[3:]
        
        # Check if there is no '-' after the first '1' beyond the initial dash
        if not re.search(r'1.*-', s_without_initial_dash):
            new_s = []
            for i, c in enumerate(s):
                if (i == 9 or i == 12) and c == '1':  # 0-based indexing
                    new_s.append('|')
                else:
                    new_s.append(c)
            return ''.join(new_s)
        else:
            return s  # return original string if there is another '-' after the initial one

    def _postprocess(self, text, patterns):
        try:
            log_and_print("Performing postprocessing", level=logging.DEBUG)
            # Replace characters in the text
            for key, value in REPLACEMENTS.items():
                text = text.replace(key, value)

            matched_text = []
            for pattern in patterns:
                matches = re.findall(pattern, text)
                matched_text.extend(matches)

            filtered_text = "\n".join(matched_text)
            # Replace "1" based on the position
            filtered_text = self.replace_10th_and_13th_char_if_no_dash(filtered_text)

            # Replace "|" with ""
            filtered_text = filtered_text.replace("|", "")

            return filtered_text
        except Exception as e:
            log_and_print(f"An error occurred: {e}", level=logging.ERROR)
            return None
        
    def visualize_sections(self, sections, text):
        try:
            for idx, (section, section_text) in enumerate(zip(sections, text), start=1):
                plt.figure(figsize=(8, 6))

                # Resize and display the section
                resized_section = section.resize((200, 150))
                plt.imshow(resized_section, cmap='gray')

                plt.title(f"Section {idx}: {section_text}")
                plt.axis('off')

                plt.show()
        except Exception as e:
            log_and_print(f"An error occurred while visualizing sections: {e}", level=logging.ERROR)

    def OCR_on_directory(self, input_dir, patterns, section_size_percentage, overlap_percentage):
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(".tif"):
                input_path = os.path.join(input_dir, filename)
                filtered_text = self.ocr_sliding_window(input_path, patterns=patterns, section_size_percentage=section_size_percentage, overlap_percentage=overlap_percentage)

                if filtered_text:
                    log_and_print(f"Filtered text: {filtered_text}", level=logging.INFO)
                    # Rename the input file to the filtered text
                    os.rename(input_path, os.path.join(input_dir, f"{filtered_text}.tif"))
                else:
                    log_and_print("No filtered text found.", level=logging.INFO)


def main():
    ocr_scan = OCRScan()
    ocr_scan.OCR_on_directory(INPUT_DIR, PATTERNS, SECTION_SIZE_PERCENTAGE, OVERLAP_PERCENTAGE)

if __name__ == "__main__":
    main()