import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "sample_videos/video_01.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully")
print("Dwell time calculation started...")

# ROI coordinates
roi_x1 = 150
roi_y1 = 100
roi_x2 = 500
roi_y2 = 400

# Store entry time for each person
entry_times = {}

# Store previous ROI status
person_status = {}

# Get video FPS
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Current video time in seconds
    current_time = frame_number / fps

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

        for box, person_id in zip(boxes.xyxy, track_ids):

            x1, y1, x2, y2 = map(int, box.tolist())

            # Calculate center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check if person is inside ROI
            inside_roi = (
                roi_x1 <= center_x <= roi_x2
                and
                roi_y1 <= center_y <= roi_y2
            )

            previous = person_status.get(person_id, False)

            # ENTRY
            if inside_roi and not previous:

                entry_times[person_id] = current_time

                print(
                    f"ENTRY: Person ID {person_id} "
                    f"at {current_time:.2f} seconds"
                )

            # EXIT
            elif not inside_roi and previous:

                if person_id in entry_times:

                    entry_time = entry_times[person_id]

                    dwell_time = current_time - entry_time

                    print(
                        f"EXIT: Person ID {person_id} "
                        f"at {current_time:.2f} seconds"
                    )

                    print(
                        f"DWELL TIME: Person ID {person_id} "
                        f"stayed {dwell_time:.2f} seconds"
                    )

                    # Remove entry time after exit
                    del entry_times[person_id]

            # Save current status
            person_status[person_id] = inside_roi

            # Draw person
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

            # Display ID
            status = "INSIDE" if inside_roi else "OUTSIDE"

            cv2.putText(
                frame,
                f"ID: {person_id} {status}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    # Show video
    cv2.imshow(
        "Intelligent Security Monitor - Dwell Time",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Dwell time calculation completed.")