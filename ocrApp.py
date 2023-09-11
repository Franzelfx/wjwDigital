import re
import os
import sys
import PIL
import logging
import pytesseract
from PIL import Image

# Log config to file
logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

PIL.Image.MAX_IMAGE_PIXELS = 933120000

INPUT_FILE = "./MFLK_Ohne Hollerith TEST UHDE/11_000001_300 dpi/Scan_0002.tif"
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
        log_and_print("Initializing OCRScan", level=logging.DEBUG)

    def _preprocess_image(self, image_path_or_obj):
        # If it's a string path, open it, otherwise assume it's an Image object
        if isinstance(image_path_or_obj, str):
            image = Image.open(image_path_or_obj)
        else:
            image = image_path_or_obj
        # Convert to grayscale
        image = image.convert("L")
        return image

    def _ocr_image(self, image_obj=None):
        try:
            custom_config = f"-c tessedit_char_whitelist={CHAR_WHITELIST}"
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

            section.save(output_path)
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
        section_size_percentage=70,
        overlap_percentage=30,
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
            section_width = int(image_width * section_size_percentage / 100)
            section_height = int(image_height * section_size_percentage / 100)
            shift_width = section_width - int(image_width * overlap_percentage / 100)
            shift_height = section_height - int(image_height * overlap_percentage / 100)

            results = []

            for y in range(0, image_height, shift_height):
                for x in range(0, image_width, shift_width):
                    right = min(x + section_width, image_width)
                    bottom = min(y + section_height, image_height)
                    section = image.crop((x, y, right, bottom))

                    text = self._ocr_on_section(
                        section,
                        x,
                        y,
                        self,
                        patterns,
                        output_folder,
                        md_file
                    )
                    results.append(text)
            # Close the Markdown file
            md_file.close()
            # Return the first non-empty or non-None result
            for result in results:
                if result:
                    return result
        except Exception as e:
            log_and_print(f"An error occurred: {e}", level=logging.ERROR)
            return None

import os
import sys
import logging
import concurrent.futures
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal

class OCRApplication(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.displayLogoImage()
        self.initOCR()

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('OCR Application')

        self.log_text_edit = QtWidgets.QTextEdit()
        self.log_text_edit.setReadOnly(True)

        self.select_dir_button = QtWidgets.QPushButton('Select Directory')
        self.select_dir_button.clicked.connect(self.selectDirectory)

        self.run_ocr_button = QtWidgets.QPushButton('Run OCR')
        self.run_ocr_button.clicked.connect(self.runOCR)
        self.run_ocr_button.setEnabled(False)

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.select_dir_button)
        layout.addWidget(self.run_ocr_button)
        layout.addWidget(self.log_text_edit)

        central_widget.setLayout(layout)

    def displayLogoImage(self):
        logo_image_path = "logo.png"  # Use the relative path to the logo image
        if os.path.exists(logo_image_path):
            logo_label = QtWidgets.QLabel(self)
            pixmap = QPixmap(logo_image_path)

            scaled_pixmap = pixmap.scaled(600, 200, QtCore.Qt.KeepAspectRatio)

            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(QtCore.Qt.AlignCenter)

            central_widget = QtWidgets.QWidget(self)
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(logo_label)
            layout.addWidget(self.select_dir_button)
            layout.addWidget(self.run_ocr_button)
            layout.addWidget(self.log_text_edit)

            central_widget.setLayout(layout)
            self.setCentralWidget(central_widget)


    def initOCR(self):
        self.ocr_thread = OCRThread()

        # Connect signals and slots for communication between the main app and the OCR thread
        self.ocr_thread.log_signal.connect(self.logMessage)
        self.ocr_thread.completed_signal.connect(self.ocrCompleted)

    def logMessage(self, message, level=logging.INFO):
        log_entry = f"{logging.getLevelName(level)} - {message}"
        logging.log(level, log_entry)
        self.log_text_edit.append(log_entry)

    def selectDirectory(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly

        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", options=options)

        if directory:
            self.selected_directory = directory
            self.logMessage(f"Selected directory: {directory}")
            self.run_ocr_button.setEnabled(True)

    def runOCR(self):
        if hasattr(self, 'selected_directory'):
            self.logMessage("Starting OCR process...", logging.INFO)
            self.run_ocr_button.setEnabled(False)
            self.select_dir_button.setEnabled(False)

            # Start the OCR thread
            self.ocr_thread.set_directory(self.selected_directory)
            self.ocr_thread.start()
        else:
            self.logMessage("Please select a directory first.", logging.WARNING)
    
    def closeEvent(self, event):
        # Ensure that the OCR thread is stopped when the app is closed
        if hasattr(self, 'ocr_thread'):
            self.ocr_thread.stopProcessing()
            self.ocr_thread.wait()  # Wait for the thread to finish gracefully
        event.accept()

    def ocrCompleted(self):
        self.logMessage("OCR process completed.", logging.INFO)
        self.run_ocr_button.setEnabled(True)
        self.select_dir_button.setEnabled(True)

class OCRThread(QThread):
    log_signal = pyqtSignal(str, int)
    completed_signal = pyqtSignal()

    def set_directory(self, directory):
        self.directory = directory

    def run(self):
        if hasattr(self, 'directory'):
            self.ocr_on_directory(self.directory)
            self.completed_signal.emit()
        else:
            self.log_signal.emit("Directory not found.", logging.ERROR)

    def ocr_on_directory(self, directory):
        if not os.path.exists(directory):
            self.log_signal.emit(f"Directory not found: {directory}", logging.ERROR)
            return

        ocr_scan = OCRScan()

        # Create a list of image paths to process
        image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.lower().endswith((".tif", ".tiff"))]

        # Define a function for processing an image
        def process_image(image_path):
            self.log_signal.emit(f"Starting OCR scan on file: {image_path}", logging.INFO)
            filtered_text = ocr_scan.ocr_on_image(image_path)
            if filtered_text:
                self.log_signal.emit(f"Filtered text: {filtered_text}", logging.INFO)
                os.rename(image_path, os.path.join(directory, f"{filtered_text}.tif"))
            else:
                self.log_signal.emit(f"No filtered text found in image: {image_path}", logging.WARNING)
                # Append "_Fehler" to the filename if no text is found
                os.rename(image_path, os.path.join(directory, f"{os.path.splitext(os.path.basename(image_path))[0]}_Fehler.tif"))

        # Get number of CPU cores
        num_cores = os.cpu_count()
        # Use ThreadPoolExecutor to parallelize image processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            executor.map(process_image, image_paths)
    
    def stopProcessing(self):
        # Custom method to stop the processing gracefully
        self.terminate()  # Terminate the thread

def main():
    app = QtWidgets.QApplication(sys.argv)
    ocr_app = OCRApplication()
    ocr_app.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    logging.basicConfig(filename="debug.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
