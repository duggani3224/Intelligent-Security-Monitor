import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Video path
video_path = "sample_videos/video_01.mp4"
video_path = "sample_videos/video_02.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully")
print("ROI monitoring started...")

# ROI coordinates
# Format: x1, y1, x2, y2
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

    # Draw ROI rectangle
    cv2.rectangle(
        frame,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "ROI",
        (roi_x1, roi_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Process tracked people
    boxes = results[0].boxes

    if boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes.xyxy, track_ids):

            x1, y1, x2, y2 = map(int, box.tolist())

            # Find center of person
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check whether center is inside ROI
            inside_roi = (
                roi_x1 <= center_x <= roi_x2
                and
                roi_y1 <= center_y <= roi_y2
            )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (255, 0, 0),
                -1
            )

            # Display tracking ID and ROI status
            status = "INSIDE ROI" if inside_roi else "OUTSIDE ROI"

            cv2.putText(
                frame,
                f"ID: {track_id} - {status}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    # Display video
    cv2.imshow(
        "Intelligent Security Monitor - ROI",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("ROI monitoring completed.")