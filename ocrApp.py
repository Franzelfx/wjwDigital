import os
import sys
import logging
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore, QtWidgets
from ocr import OCRScan, resource_path
from PyQt5.QtCore import QThread, pyqtSignal
import queue
import traceback

def get_unique_filename(target_path):
    base, ext = os.path.splitext(target_path)
    counter = 1
    while os.path.exists(target_path):
        target_path = f"{base}_{counter}{ext}"
        counter += 1
    return target_path

class LogThread(QThread):
    log_signal = pyqtSignal(str, int)

    def __init__(self):
        super().__init__()
        self.log_queue = queue.Queue()  # Create a queue to store log messages

    def run(self):
        while True:
            message, level = self.log_queue.get()  # Get a log message from the queue
            if message is None:
                break
            log_entry = f"{logging.getLevelName(level)} - {message}"
            logging.log(level, log_entry)

            # Emit a signal to update the GUI with the log message
            self.log_signal.emit(log_entry, level)

    def stop(self):
        # Stop the thread by adding a None message to the queue
        self.log_queue.put((None, None))

class OCRThread(QThread):
    # Define signals to update the GUI
    progress_signal = pyqtSignal(str)  # Signal to update progress
    result_signal = pyqtSignal(str)    # Signal to update results

    def __init__(self, directory, run_again_with_enhanced_image=True, confidence_threshold=5):
        super().__init__()
        self.directory = directory
        self.confidence_threshold = confidence_threshold
        self.run_again_with_enhanced_image = run_again_with_enhanced_image

    def run(self):
        # If we are on windows exe path for tesseract is: C:\Program Files\Tesseract-OCR\tesseract.exe
        if sys.platform == "win32":
            path = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            ocr_scan = OCRScan(tesseract_path=path, confidence_threshold=self.confidence_threshold)
        else:
            ocr_scan = OCRScan(confidence_threshold=self.confidence_threshold)
        image_paths = []
        for root, dirs, files in os.walk(self.directory):
            for filename in files:
                if filename.lower().endswith(('.tif', '.tiff')):
                    image_paths.append(os.path.join(root, filename))

        for image_path in image_paths:
            if not self.isRunning:
                break
            self.progress_signal.emit(f"Processing image: {image_path}")
            filtered_text = ocr_scan.ocr_on_image(image_path, run_again_with_enhanced_image=self.run_again_with_enhanced_image)

            original_dir = os.path.dirname(image_path)  # Get the directory of the current image
            if filtered_text:
                self.result_signal.emit(f"Filtered text: {filtered_text}")
                target_path = os.path.join(original_dir, f"{filtered_text}")
                unique_target_path = get_unique_filename(target_path)
                # Rename if not already exist
                os.rename(image_path, unique_target_path)
                # Append _OCR-korrekt to the filename
                os.rename(unique_target_path, f"{unique_target_path}_OCR-korrekt.tif")
                self.result_signal.emit(f"Renamed image to: {unique_target_path}_OCR-korrekt.tif")
            else:
                original_basename = os.path.basename(image_path)[:-4] # Get original basename without extension
                
                # Only append "_Fehler" if it's not already in the filename
                if "_Fehler" not in original_basename:
                    target_name = f"{original_basename}_Fehler.tif"
                else:
                    target_name = f"{original_basename}.tif"

                if "_Hollerith" not in original_basename:
                    target_path = os.path.join(original_dir, target_name)
                    unique_target_path = get_unique_filename(target_path)
                    os.rename(image_path, unique_target_path)
                    self.result_signal.emit(f"No filtered text found in image: {image_path}")
                    self.result_signal.emit(f"Renamed image to: {unique_target_path}")
                else:
                    self.result_signal.emit(f"Image already scanned: {image_path}")

    def stopProcessing(self):
        self._isRunning = False

class OCRApplication(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.displayLogoImage()

        self.log_thread = LogThread()
        self.log_thread.log_signal.connect(self.updateLogText)
        self.log_thread.start()

    def logMessage(self, message, level=logging.INFO):
        # Add log message to the queue in the logging thread
        self.log_thread.log_queue.put((message, level))

    def updateLogText(self, log_entry):
        # Update the GUI with the log message
        self.log_text_edit.append(log_entry)

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('OCR Application')
        self.log_text_edit = QtWidgets.QTextEdit()
        self.log_text_edit.setReadOnly(True)

        self.run_again_with_enhanced_image_checkbox = QtWidgets.QCheckBox("Run again with enhanced image")
        self.run_again_with_enhanced_image_checkbox.setChecked(True)

        self.select_dir_button = QtWidgets.QPushButton('Select Directory')
        self.select_dir_button.clicked.connect(self.selectDirectory)

        self.run_ocr_button = QtWidgets.QPushButton('Run OCR')
        self.run_ocr_button.clicked.connect(self.runOCR)
        self.run_ocr_button.setEnabled(False)

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        self.confidence_threshold_label = QtWidgets.QLabel('Confidence Threshold [%]')
        self.confidence_threshold_edit = QtWidgets.QLineEdit()
        self.confidence_threshold_edit.setPlaceholderText('Enter confidence threshold (0-100)')
        self.confidence_threshold_edit.setText('5')


        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.select_dir_button)
        layout.addWidget(self.run_ocr_button)
        layout.addWidget(self.log_text_edit)
        layout.addWidget(self.run_again_with_enhanced_image_checkbox)
        layout.addWidget(self.confidence_threshold_label)
        layout.addWidget(self.confidence_threshold_edit)

        central_widget.setLayout(layout)

    def displayLogoImage(self):
        logo_image_path = resource_path("logo.png")
        if os.path.exists(logo_image_path):
            logo_label = QtWidgets.QLabel(self)
            pixmap = QPixmap(logo_image_path)

            scaled_pixmap = pixmap.scaled(600, 200, QtCore.Qt.KeepAspectRatio)

            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(QtCore.Qt.AlignCenter)
            
            central_widget = self.centralWidget()
            central_layout = central_widget.layout()
            central_layout.insertWidget(0, logo_label)

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
            
            directory = self.selected_directory
            confidence_threshold = float(self.confidence_threshold_edit.text() or 5)
            run_again_with_enhanced_image = self.run_again_with_enhanced_image_checkbox.isChecked()
            self.ocr_thread = OCRThread(directory, run_again_with_enhanced_image, confidence_threshold)
            self.ocr_thread.progress_signal.connect(self.logMessage)
            self.ocr_thread.result_signal.connect(self.logMessage)
            self.ocr_thread.finished.connect(self.ocrCompleted)
            self.ocr_thread.start()
    
    def closeEvent(self, event):
        # Ensure that the OCR thread is stopped when the app is closed
        if hasattr(self, 'ocr_thread') and self.ocr_thread.isRunning():
            self.ocr_thread.stopProcessing()
            self.ocr_thread.wait()
        event.accept()

    def ocrCompleted(self):
        self.logMessage("OCR process completed.", logging.INFO)
        self.run_ocr_button.setEnabled(True)
        self.select_dir_button.setEnabled(True)

def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        ocr_app = OCRApplication()
        ocr_app.show()
        sys.exit(app.exec_())
    except Exception as e:
        # Logg with traceback
        logging.error(traceback.format_exc())

if __name__ == '__main__':
    logging.basicConfig(filename="debug.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
