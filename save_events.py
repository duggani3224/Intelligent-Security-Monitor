import cv2
import csv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "sample_videos/video_01.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully")
print("Event recording started...")

# ROI coordinates
roi_x1 = 150
roi_y1 = 100
roi_x2 = 500
roi_y2 = 400

# Get FPS
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

frame_number = 0

# Store previous status
person_status = {}

# Store entry time
entry_times = {}

# Store events
events = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    current_time = frame_number / fps

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

    boxes = results[0].boxes

    if boxes.id is not None:

        track_ids = boxes.id.int().cpu().tolist()

        for box, person_id in zip(boxes.xyxy, track_ids):

            x1, y1, x2, y2 = map(int, box.tolist())

            # Center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check ROI
            inside_roi = (
                roi_x1 <= center_x <= roi_x2
                and
                roi_y1 <= center_y <= roi_y2
            )

            previous = person_status.get(person_id, False)

            # ENTRY
            if inside_roi and not previous:

                entry_times[person_id] = current_time

                events.append({
                    "track_id": person_id,
                    "event": "ENTRY",
                    "frame": frame_number,
                    "timestamp_s": round(current_time, 2),
                    "entry_time_s": round(current_time, 2),
                    "exit_time_s": "",
                    "duration_s": "",
                    "status": "INSIDE"
                })

                print(f"ENTRY: Person ID {person_id}")

            # EXIT
            elif not inside_roi and previous:

                if person_id in entry_times:

                    entry_time = entry_times[person_id]

                    dwell_time = current_time - entry_time

                    events.append({
                        "track_id": person_id,
                        "event": "EXIT",
                        "frame": frame_number,
                        "timestamp_s": round(current_time, 2),
                        "entry_time_s": round(entry_time, 2),
                        "exit_time_s": round(current_time, 2),
                        "duration_s": round(dwell_time, 2),
                        "status": "OUTSIDE"
                    })

                    print(
                        f"EXIT: Person ID {person_id} "
                        f"Dwell: {dwell_time:.2f} seconds"
                    )

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

            cv2.putText(
                frame,
                f"ID: {person_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    cv2.imshow(
        "Intelligent Security Monitor - Events",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Create CSV file
csv_file = "events.csv"

with open(csv_file, "w", newline="") as file:

    fieldnames = [
        "track_id",
        "event",
        "frame",
        "timestamp_s",
        "entry_time_s",
        "exit_time_s",
        "duration_s",
        "status"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(events)

print("Event recording completed.")
print(f"CSV file created: {csv_file}")