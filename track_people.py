import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Video path
video_path = "sample_videos/video_01.mp4"
video_path = "sample_videos/video_02.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully")
print("People tracking started...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Track only people
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.5,
        verbose=False
    )

    # Draw bounding boxes and tracking IDs
    annotated_frame = results[0].plot()

    # Display video
    cv2.imshow(
        "Intelligent Security Monitor - Tracking",
        annotated_frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("People tracking completed.")