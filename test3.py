import time
import cv2
import pygetwindow as gw
import threading
import os
import datetime
import pyautogui
from ultralytics import YOLO
from fpdf import FPDF
from PyPDF2 import PdfWriter

target_title_substr = "LinkedIn"  # Adjust this to a substring of the window title
stop_flag = threading.Event()  # Create a threading event for stopping
cap = None  # Initialize the webcam variable globally
cap_lock = threading.Lock()  # Lock for thread-safe access to cap
detection_entries = []
output_dir = r"C:\Users\vipas\Phone-detection\output"  # Specify your desired directory
snapshot_dir = os.path.join(output_dir, "snapshots")  # Directory for saving snapshots
pdf_path = os.path.join(output_dir, "detection_log.pdf")  # Path for the PDF file
last_saved_second = None
manual_stop = False  # Flag to track manual stop

# Load YOLO model
model = YOLO('yolov8m.pt')  # Choose the appropriate model size

def create_pdf(detections, pdf_path):
    # Remove the existing PDF file if it exists
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    
    pdf_writer = PdfWriter()

    # Add new content to the PDF
    pdf_temp = FPDF()
    pdf_temp.add_page()
    pdf_temp.set_font("Arial", size=12)
    pdf_temp.cell(200, 10, txt="Detection Log", ln=True, align="C")
    pdf_temp.ln(10)  # Line break

    image_width = 180
    image_height = 90
    page_height = pdf_temp.h - 20  # Reduce margin from the page height to avoid printing over the bottom margin

    for entry in detections:
        timestamp, image_paths = entry
        pdf_temp.set_text_color(255, 0, 0)
        pdf_temp.cell(200, 10, txt=timestamp, ln=True, align="L")
        pdf_temp.ln(5)  # Add space below the timestamp

        for image_path in image_paths:
            if image_path:
                # Calculate the position where the image would end
                image_end_y = pdf_temp.get_y() + image_height

                # Check if adding the image would go beyond the page's bottom margin
                if image_end_y > page_height:
                    pdf_temp.add_page()  # Start a new page if it doesn't fit

                try:
                    pdf_temp.image(image_path, x=10, y=pdf_temp.get_y(), w=image_width, h=image_height)
                except RuntimeError as e:
                    print(f"Error adding image to PDF: {e}")
                    pdf_temp.add_page()  # Add a new page if an error occurs

                pdf_temp.ln(image_height + 10)  # Move cursor to the next line after the image

        pdf_temp.ln(10)  # Add space after each entry

    # Save the temporary PDF
    pdf_temp.output(pdf_path)

    print(f"Detection log updated at: {pdf_path}")  # Print the log location
    return pdf_path

def detect_phone_and_humans():
    global cap, detection_entries, last_saved_second, stop_flag

    while not stop_flag.is_set():
        with cap_lock:  # Ensure thread-safe access to cap
            if cap is None or not cap.isOpened():
                time.sleep(1)
                continue

            ret, frame = cap.read()

        if not ret:
            break

        # Create a blurred version of the frame for background blur
        blurred_frame = cv2.GaussianBlur(frame, (71, 71), 0)

        person_counter = 0  # Reset person counter for each frame
        detected_phone = False
        image_paths = []
        human_count = 0

        results = model(frame)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            names = result.names

            for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                x1, y1, x2, y2 = map(int, box)
                label = names[int(cls)]

                if label == 'cell phone' and conf > 0.5:
                    detected_phone = True
                    color = (0, 255, 0)  # Green for cell phones
                    # Overlay detected phone on the blurred background
                    blurred_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                    cv2.rectangle(blurred_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(blurred_frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                elif label == 'person':
                    human_count += 1
                    person_counter += 1  # Increment person counter
                    person_label = f'Person {person_counter}'
                    color = (255, 0, 0)  # Blue for humans
                    # Overlay detected person on the blurred background
                    blurred_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                    cv2.rectangle(blurred_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(blurred_frame, f'{person_label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Capture screenshots of both the detection and the entire screen if phone or multiple people are detected
        if detected_phone or human_count > 1:
            # Save the blurred frame with bounding boxes
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            blurred_frame_path = os.path.join(snapshot_dir, f"blurred_frame_{timestamp}.png")
            os.makedirs(snapshot_dir, exist_ok=True)
            cv2.imwrite(blurred_frame_path, blurred_frame)

            # Capture a screenshot of the entire screen
            screenshot_path = os.path.join(snapshot_dir, f"screenshot_{timestamp}.png")
            pyautogui.screenshot(screenshot_path)

            detection_entries.append((f"Phone or multiple humans detected at: {timestamp}", [blurred_frame_path, screenshot_path]))
            last_saved_second = timestamp

        # Show the frame with blurred background and sharp detected objects
        cv2.imshow("Phone and Human Detection", blurred_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()

    with cap_lock:  # Release the webcam safely
        if cap is not None:
            cap.release()
    cv2.destroyAllWindows()

def detect_target_window():
    global cap, stop_flag, detection_entries, manual_stop
    while True:
        windows = gw.getWindowsWithTitle(target_title_substr)
        if windows:
            target_window = windows[0]
            if target_window.isMinimized or not target_window.isActive:
                with cap_lock:  # Ensure thread-safe access to cap
                    if cap is not None and cap.isOpened():
                        print("Target window is minimized or not active. Stopping the webcam...")
                        cap.release()
                        cap = None
                        stop_flag.set()  # Stop the detection thread
                        create_pdf(detection_entries, pdf_path)  # Generate the PDF when the webcam stops
                        detection_entries = []  # Clear the entries after saving to PDF
                        manual_stop = False  # Reset manual stop flag
            else:
                with cap_lock:  # Ensure thread-safe access to cap
                    if cap is None or not cap.isOpened():
                        print("Target window detected and active! Starting the webcam...")
                        cap = cv2.VideoCapture(0)
                        if cap.isOpened():
                            stop_flag.clear()  # Clear the stop flag before starting
                            threading.Thread(target=detect_phone_and_humans).start()
                        else:
                            print("Error: Failed to open webcam.")
        else:
            with cap_lock:  # Ensure thread-safe access to cap
                if cap is not None and cap.isOpened():
                    print("Target window closed. Stopping the webcam...")
                    cap.release()
                    cap = None
                    stop_flag.set()  # Stop the detection thread
                    create_pdf(detection_entries, pdf_path)  # Generate the PDF when the webcam stops
                    detection_entries = []  # Clear the entries after saving to PDF
                    manual_stop = False  # Reset manual stop flag

        time.sleep(2)

if __name__ == "__main__":
    print("Starting Detection")
    try:
        # Ensure the PDF is cleared before starting a new run
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        detect_target_window()
    except KeyboardInterrupt:
        print("Program stopped manually.")
        manual_stop = True
    finally:
        # Ensure resources are cleaned up properly
        with cap_lock:
            if cap is not None:
                cap.release()
        stop_flag.set()
        if not manual_stop:  # Create PDF only if not manually stopped
            create_pdf(detection_entries, pdf_path)  # Create the final PDF
