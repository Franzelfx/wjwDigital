import re
import os
import sys
import PIL
from PIL import Image
import logging
import pytesseract
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtCore import QRunnable, QThreadPool, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QTextEdit,
)
import traceback
from PIL import ImageFilter, ImageEnhance
import cv2
import numpy as np

# Log config to file
logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

PIL.Image.MAX_IMAGE_PIXELS = 933120000

INPUT_DIR = "./MFLK_Ohne Hollerith TEST UHDE/11_000001_300 dpi"
OUTPUT_DIR = f"{INPUT_DIR}/results"
OVERLAP_PERCENTAGE = 20
SECTION_SIZE_PERCENTAGE = 80
PATTERNS = [
    r"\d{2}-\w{10}",
    r"\d{2}-\d+-\d{2}-\d"
]
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
CHAR_WHITELIST = "0123456789A-"

def log_and_print(message, level=logging.INFO, file_only=False, qt_text_edit=None):
    log_entry = f"{logging.getLevelName(level)} - {message}"
    logging.log(level, log_entry)
    if level >= logging.ERROR and not file_only:
        print(message, file=sys.stderr)
    elif not file_only:
        print(message)
    if qt_text_edit:
        qt_text_edit.append(log_entry)

# Auto-detect Tesseract path
if sys.platform.startswith('win32'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
elif sys.platform.startswith('darwin'):
    # For macOS, the path usually will be /usr/local/bin/tesseract if installed via brew
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
elif sys.platform.startswith('linux'):
    # For Linux, it's usually just 'tesseract'
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
else:
    log_and_print("Unsupported OS", level=logging.ERROR)
    sys.exit(1)    

class OCRScan:
    def __init__(self):
        self._is_running = True
        log_and_print("Initializing OCRScan", level=logging.DEBUG)
        # Configuration for pytesseract autodetect pytesseract path on Windows or mac

    def preprocess_image(self, image_path_or_obj):
        # If it's a string path, open it, otherwise assume it's an Image object
        if isinstance(image_path_or_obj, str):
            image = Image.open(image_path_or_obj)
        else:
            image = image_path_or_obj
        # Convert to grayscale
        image = image.convert("L")
        return image

    def ocr_image(self, image_obj=None):
        try:
            custom_config = f"-c tessedit_char_whitelist={CHAR_WHITELIST}"
            image_obj = self.preprocess_image(image_obj)
            text = pytesseract.image_to_string(image_obj, config=custom_config)
            return text
        except Exception as e:
            log_and_print(f"An error occurred: {e}", level=logging.ERROR)
            return None


    # Worker function to be run in parallel
    def ocr_on_section(
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

            section.save(output_path)
            section_text = ocr_instance.ocr_image(image_obj=section)

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

    def ocr_sliding_window(
        self,
        image_path,
        patterns=PATTERNS,
        section_size_percentage=70,
        overlap_percentage=30,
    ):
        try:
            log_and_print(
                f"Performing OCR with sliding window on image {image_path}",
                level=logging.DEBUG,
            )

            # Create a dynamic output folder based on the input file name
            base_filename = os.path.basename(image_path).split(".")[0]
            output_folder = os.path.join(OUTPUT_DIR, base_filename, "sections")
            # Create a log file to store the OCR results
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Create a markdown file to store the OCR results
            md_file_path = os.path.join(OUTPUT_DIR, base_filename, "OCR_Results.md")
            md_file = open(md_file_path, "w")

            image = Image.open(image_path)
            image_width, image_height = image.size
            section_width = int(image_width * section_size_percentage / 100)
            section_height = int(image_height * section_size_percentage / 100)
            shift_width = section_width - int(image_width * overlap_percentage / 100)
            shift_height = section_height - int(image_height * overlap_percentage / 100)

            results = []
            file_data = []  # List to store file data for CSV

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []

                for y in range(0, image_height, shift_height):
                    for x in range(0, image_width, shift_width):
                        right = min(x + section_width, image_width)
                        bottom = min(y + section_height, image_height)
                        section = image.crop((x, y, right, bottom))

                        # Fügen Sie die Tasks dem ThreadPoolExecutor hinzu
                        futures.append(
                            executor.submit(
                                self.ocr_on_section,
                                section,
                                x,
                                y,
                                self,
                                patterns,
                                output_folder,
                                md_file
                            )
                        )

                # Ergebnisse sammeln
                results = [f.result() for f in futures if f.result()]

            # Schließen Sie die Markdown-Datei
            md_file.close()
            # Return only one result
            return results[0] if results else None
        except Exception as e:
            log_and_print(f"An error occurred: {e}", level=logging.ERROR)
            return None

    def _postprocess(self, text, patterns):
        try:
            matched_text = []
            for pattern in patterns:
                matches = re.findall(pattern, text)
                matched_text.extend(matches)
            filtered_text = matched_text[0] if matched_text else None
            return filtered_text
        except Exception as e:
            log_and_print(f"An error occurred: {e}", level=logging.ERROR)
            # Print traceback
            traceback.print_exc()
            return None

    def OCR_on_directory(
        self,
        directory,
        patterns=PATTERNS,
        section_size_percentage=70,
        overlap_percentage=30,
        qt_text_edit=None,
    ):
        log_and_print(
            f"OCR scan directory: {directory}",
            level=logging.INFO,
            qt_text_edit=qt_text_edit,
        )
        for filename in os.listdir(directory):
            if filename.lower().endswith(".tif"):
                input_path = os.path.join(directory, filename)
                log_and_print(
                    f"Starting OCR scan on file: {filename}",
                    level=logging.INFO,
                    qt_text_edit=qt_text_edit,
                )
                filtered_text = self.ocr_sliding_window(
                    input_path,
                    patterns=patterns,
                    section_size_percentage=section_size_percentage,
                    overlap_percentage=overlap_percentage,
                )
                if filtered_text:
                    log_and_print(
                        f"Filtered text: {filtered_text}",
                        level=logging.INFO,
                        qt_text_edit=qt_text_edit,
                    )
                    # Rename the input file to the filtered text
                    os.rename(
                        input_path, os.path.join(directory, f"{filtered_text}.tif")
                    )
                else:
                    log_and_print(
                        "No filtered text found.",
                        level=logging.INFO,
                        qt_text_edit=qt_text_edit,
                    )
            if not self._is_running:
                log_and_print(
                    "OCR scan stopped.", level=logging.INFO, qt_text_edit=qt_text_edit
                )
                break
        log_and_print(
            "Completed OCR scan on directory.",
            level=logging.INFO,
            qt_text_edit=qt_text_edit,
        )


class WorkerSignals(QObject):
    finished = pyqtSignal()


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.fn(*self.args, **self.kwargs)


class MyOCRApp(QMainWindow):
    def __init__(self, ocr_scan):
        super().__init__()

        self.ocr_scan = ocr_scan
        self.threadpool = QThreadPool()

        self.setWindowTitle("AI OCR App")
        self.setGeometry(200, 200, 1000, 500)

        layout = QVBoxLayout()

        # Create QTextEdit for logs
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)

        # Create Choose Directory button
        self.choose_directory_button = QPushButton("Choose Directory")
        self.choose_directory_button.clicked.connect(self.choose_directory)

        layout.addWidget(self.log_window)
        layout.addWidget(self.choose_directory_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def choose_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            worker = Worker(
                self.ocr_scan.OCR_on_directory,
                directory,
                # Other parameters,
                qt_text_edit=self.log_window,  # Pass the QTextEdit to log_and_print
                patterns=PATTERNS,
                section_size_percentage=SECTION_SIZE_PERCENTAGE,
                overlap_percentage=OVERLAP_PERCENTAGE,
            )
            self.threadpool.start(worker)

    def closeEvent(self, event):
        """Overridden function to stop OCR process when the window is closed."""
        self.ocr_scan.is_running = False  # Set the flag to False
        event.accept()  # Close the application


def main():
    # app = QApplication(sys.argv)
    ocr_scan = OCRScan()
    ocr_scan.OCR_on_directory(
        INPUT_DIR,
        patterns=PATTERNS,
        section_size_percentage=SECTION_SIZE_PERCENTAGE,
        overlap_percentage=OVERLAP_PERCENTAGE,
    )
    # my_app = MyOCRApp(ocr_scan)
    # my_app.show()
    # app.exec_()


if __name__ == "__main__":
    main()
