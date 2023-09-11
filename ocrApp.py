import os
import sys
import logging
from PyQt5 import QtCore, QtWidgets
from TrOCR.ocr import OCRScan
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
        logo_image_path = "./image/logo.png"  # Replace with the actual path to your image
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
            # Emit the log_signal with two arguments (a string and an int)
            self.log_signal.emit("Directory not found.", logging.ERROR)

    def ocr_on_directory(self, directory):
        if not os.path.exists(directory):
            # Emit the log_signal with two arguments (a string and an int)
            self.log_signal.emit(f"Directory not found: {directory}", logging.ERROR)
            return

        ocr_scan = OCRScan()

        for filename in os.listdir(directory):
            if filename.lower().endswith((".tif", ".tiff")):
                input_path = os.path.join(directory, filename)
                # Emit the log_signal with two arguments (a string and an int)
                self.log_signal.emit(f"Starting OCR scan on file: {filename}", logging.INFO)
                filtered_text = ocr_scan.ocr_on_image(input_path)
                if filtered_text:
                    # Emit the log_signal with two arguments (a string and an int)
                    self.log_signal.emit(f"Filtered text: {filtered_text}", logging.INFO)
                    # Rename the input file to the filtered text
                    os.rename(input_path, os.path.join(directory, f"{filtered_text}.tif"))
                else:
                    # Emit the log_signal with two arguments (a string and an int)
                    self.log_signal.emit("No filtered text found.", logging.INFO)
                    # Rename the input file to indicate an error
                    os.rename(input_path, os.path.join(directory, "_Error.tif"))

def main():
    app = QtWidgets.QApplication(sys.argv)
    ocr_app = OCRApplication()
    ocr_app.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    logging.basicConfig(filename="debug.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
