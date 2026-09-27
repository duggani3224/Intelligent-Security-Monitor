import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Video path
video_path = "sample_videos/video_01.mp4"

# Test configurations
image_sizes = [480, 640]
frame_skips = [1, 2]

print("====================================")
print(" Intelligent Security Monitor")
print(" Performance Testing")
print("====================================")

for image_size in image_sizes:

    for frame_skip in frame_skips:

        print()
        print("------------------------------------")
        print(f"Image Size : {image_size}")
        print(f"Frame Skip : {frame_skip}")
        print("------------------------------------")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("ERROR: Could not open video")
            continue

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        processed_frames = 0
        detected_people = 0

        start_time = time.time()

        frame_number = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # Skip frames
            if frame_number % frame_skip != 0:
                continue

            # YOLO detection
            results = model(
                frame,
                imgsz=image_size,
                classes=[0],
                conf=0.5,
                verbose=False
            )

            boxes = results[0].boxes

            if boxes is not None:
                detected_people += len(boxes)

            processed_frames += 1

        end_time = time.time()

        cap.release()

        processing_time = end_time - start_time

        if processing_time > 0:
            fps = processed_frames / processing_time
        else:
            fps = 0

        print(f"Total Video Frames : {total_frames}")
        print(f"Processed Frames   : {processed_frames}")
        print(f"Processing Time    : {processing_time:.2f} seconds")
        print(f"Average FPS        : {fps:.2f}")
        print(f"People Detected    : {detected_people}")

print()
print("====================================")
print(" Performance Testing Completed")
print("====================================")