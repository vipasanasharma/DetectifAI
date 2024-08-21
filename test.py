import time
import cv2
import pygetwindow as gw
import threading
import os
import datetime
from ultralytics import YOLO
from fpdf import FPDF

target_title_substr = "WhatsApp"  # Adjust this to a substring of the window title
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
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Detection Log", ln=True, align="C")
    pdf.ln(10)  # Line break

    image_width = 180
    image_height = 90

    for entry in detections:
        timestamp, image_path = entry
        if "Phone detected" in timestamp or "Multiple humans detected" in timestamp:
            pdf.set_text_color(255, 0, 0)
        else:
            pdf.set_text_color(0, 0, 0)
        pdf.cell(200, 10, txt=timestamp, ln=True, align="L")

        if image_path:
            pdf.ln(5)
            y_position = pdf.get_y()
            try:
                pdf.image(image_path, x=10, y=y_position, w=image_width, h=image_height)
            except RuntimeError as e:
                print(f"Error adding image to PDF: {e}")
            y_position += image_height + 10
            pdf.set_y(y_position)
        else:
            pdf.ln(10)

    pdf.output(pdf_path)
    print(f"Detection log saved at: {pdf_path}")  # Print the log location
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
        image_path = None

        human_count = 0

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

        # Log phone detection and/or multiple humans
        current_time = datetime.datetime.now()
        current_second = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        if detected_phone:
            if current_second != last_saved_second:
                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                image_path = os.path.join(snapshot_dir, f"screenshot_{timestamp.replace(':', '-')}.png")
                os.makedirs(snapshot_dir, exist_ok=True)
                if cv2.imwrite(image_path, frame):
                    print(f"Phone detection image saved at: {image_path}")
                else:
                    print(f"Failed to save phone detection image at: {image_path}")
                detection_entries.append((f"Phone detected at: {timestamp}", image_path))
                last_saved_second = current_second

        if human_count > 1:
            multiple_humans_detected = True
            if current_second != last_saved_second:
                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                if not detected_phone:  # Only save image if not already saved
                    image_path = os.path.join(snapshot_dir, f"screenshot_{timestamp.replace(':', '-')}.png")
                    os.makedirs(snapshot_dir, exist_ok=True)
                    if cv2.imwrite(image_path, frame):
                        print(f"Multiple humans detection image saved at: {image_path}")
                    else:
                        print(f"Failed to save multiple humans detection image at: {image_path}")
                detection_entries.append((f"Multiple humans detected at: {timestamp}", image_path))
                last_saved_second = current_second

        if not detected_phone and not multiple_humans_detected:
            if current_second != last_saved_second:
                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                detection_entries.append((f"No phone or multiple humans detected at: {timestamp}", None))
                last_saved_second = current_second

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
            linkedIn_window = windows[0]
            if linkedIn_window.isMinimized or not linkedIn_window.isActive:
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
                        if cap is None:  # Initialize webcam if not already done
                            cap = cv2.VideoCapture(0)
                        if cap.isOpened():
                            stop_flag.clear()  # Clear the stop flag before starting
                            threading.Thread(target=detect_phone_and_humans, daemon=True).start()
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

        time.sleep(1)  # Reduce sleep time to check more frequently

if __name__ == "__main__":
    try:
        detect_target_window()
    except KeyboardInterrupt:
        print("Program stopped manually.")
        create_pdf(detection_entries, pdf_path)  # Ensure PDF is created if the program is stopped manually
