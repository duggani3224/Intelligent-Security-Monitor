import cv2
from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

# Input video
video_path = "sample_videos/video_01.mp4"

# Output video
output_path = "outputs/processed_video.mp4"

# Open input video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully")
print("Processed video creation started...")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30

# Make sure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

# ROI coordinates
roi_x1 = 150
roi_y1 = 100
roi_x2 = 500
roi_y2 = 400

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Track people
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.5,
        verbose=False
    )

    # Get annotated frame
    annotated_frame = results[0].plot()

    # Draw ROI
    cv2.rectangle(
        annotated_frame,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        "ROI",
        (roi_x1, roi_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Save frame to output video
    out.write(annotated_frame)

    # Display video
    cv2.imshow(
        "Intelligent Security Monitor - Processed Video",
        annotated_frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processed video creation completed.")
print(f"Saved video: {output_path}")