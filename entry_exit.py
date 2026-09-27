import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Video path
video_path = "sample_videos/video_01.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully")
print("Entry/Exit detection started...")

# ROI coordinates
roi_x1 = 150
roi_y1 = 100
roi_x2 = 500
roi_y2 = 400

# Store previous ROI status of each person
previous_status = {}

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

    # Draw ROI
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

    boxes = results[0].boxes

    if boxes.id is not None:

        track_ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes.xyxy, track_ids):

            x1, y1, x2, y2 = map(int, box.tolist())

            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check whether center is inside ROI
            inside_roi = (
                roi_x1 <= center_x <= roi_x2
                and
                roi_y1 <= center_y <= roi_y2
            )

            # Get previous status
            was_inside = previous_status.get(track_id, False)

            # Detect ENTRY
            if inside_roi and not was_inside:
                print(f"ENTRY: Person ID {track_id}")

            # Detect EXIT
            elif not inside_roi and was_inside:
                print(f"EXIT: Person ID {track_id}")

            # Save current status
            previous_status[track_id] = inside_roi

            # Draw person box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw center
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (255, 0, 0),
                -1
            )

            # Display ID
            status = "INSIDE" if inside_roi else "OUTSIDE"

            cv2.putText(
                frame,
                f"ID: {track_id} {status}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    # Show video
    cv2.imshow(
        "Intelligent Security Monitor - Entry Exit",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Entry/Exit detection completed.")