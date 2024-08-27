import time
import cv2
import pygetwindow as gw
import threading
import os
import datetime
from ultralytics import YOLO
from fpdf import FPDF
from PyPDF2 import PdfWriter
import numpy as np
import pyautogui  # Added for taking full screen screenshots

target_title_substr = "LinkedIn"  # Adjust this to a substring of the window title
stop_flag = threading.Event()  # Create a threading event for stopping
cap = None  # Initialize the webcam variable globally
cap_lock = threading.Lock()  # Lock for thread-safe access to cap
detection_entries = []
output_dir = r"C:\Users\vipas\Phone-detection\output"  # Specify your desired directory
snapshot_dir = os.path.join(output_dir, "snapshots")  # Directory for saving snapshots
pdf_path = os.path.join(output_dir, "detection_log.pdf")  # Path for the PDF file
last_saved_second = None

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
    margin = 10

    for entry in detections:
        timestamp, image_paths = entry  # Updated to handle multiple images
        if "Phone detected" in timestamp or "Multiple humans detected" in timestamp:
            pdf_temp.set_text_color(255, 0, 0)
        else:
            pdf_temp.set_text_color(0, 0, 0)
        pdf_temp.cell(200, 10, txt=timestamp, ln=True, align="L")

        if image_paths:
            for image_path in image_paths:
                pdf_temp.ln(5)
                
                # Check if adding the image would exceed the page height
                current_y = pdf_temp.get_y()
                if current_y + image_height + margin > pdf_temp.h - pdf_temp.b_margin:
                    pdf_temp.add_page()  # Add a new page if space is not enough
                
                try:
                    y_position = pdf_temp.get_y()
                    pdf_temp.image(image_path, x=10, y=y_position, w=image_width, h=image_height)
                    # print(f"Added image to PDF: {image_path}")  # Debugging statement
                except RuntimeError as e:
                    print(f"Error adding image to PDF: {e}")
                
                # Move the Y position down after adding an image
                pdf_temp.set_y(y_position + image_height + margin)
        else:
            pdf_temp.ln(10)

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

        person_counter = 0  # Reset person counter for each frame

        results = model(frame)
        detected_phone = False
        multiple_humans_detected = False
        image_paths = []  # Updated to handle multiple images

        human_count = 0

        # Create a mask for background blurring
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                elif label == 'person':
                    human_count += 1
                    person_counter += 1  # Increment person counter
                    person_label = f'Person {person_counter}'
                    color = (255, 0, 0)  # Blue for humans
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{person_label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # Update mask to exclude detected human from blurring
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Blur the background where the mask is not set
        blurred_frame = cv2.GaussianBlur(frame, (71, 71), 0)
        frame = np.where(mask[:, :, None] == 255, frame, blurred_frame)

        # Log phone detection and/or multiple humans
        current_time = datetime.datetime.now()
        current_second = current_time.strftime("%Y-%m-%d %H:%M:%S")

        if detected_phone or human_count > 1:
            if detected_phone:
                event_description = "Phone detected"
            if human_count > 1:
                event_description = "Multiple humans detected"
            timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
            image_path = os.path.join(snapshot_dir, f"screenshot_{timestamp.replace(':', '-')}.png")
            os.makedirs(snapshot_dir, exist_ok=True)
            cv2.imwrite(image_path, frame)
            image_paths.append(image_path)

            # Take a full-screen screenshot and save it
            screen_path = os.path.join(snapshot_dir, f"screen_{timestamp.replace(':', '-')}.png")
            pyautogui.screenshot(screen_path)
            image_paths.append(screen_path)

            detection_entries.append((f"{event_description} at: {timestamp}", image_paths))
            last_saved_second = current_second

            # Debugging statements
            # print(f"Detection: {event_description} at {timestamp}")
            # print(f"Snapshot saved: {image_path}")
            # print(f"Full screen saved: {screen_path}")

        if not detected_phone and not multiple_humans_detected:
            timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
            detection_entries.append((f"No phone or multiple humans detected at: {timestamp}", None))
            last_saved_second = current_second

            # Debugging statement
            print(f"No phone or multiple humans detected at: {timestamp}")

        cv2.imshow("Phone and Human Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()

    with cap_lock:  # Release the webcam safely
        if cap is not None:
            cap.release()
    cv2.destroyAllWindows()


def detect_target_window():
    global cap, stop_flag, detection_entries
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

if __name__ == "__main__":
    print("Starting..")
    try:
        # Ensure the PDF is cleared before starting a new run
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        detect_target_window()
    except KeyboardInterrupt:
        print("Program stopped manually.")
    finally:
        # Ensure resources are cleaned up properly
        with cap_lock:
            if cap is not None:
                cap.release()
        stop_flag.set()
        create_pdf(detection_entries, pdf_path)  # Create the final PDF
