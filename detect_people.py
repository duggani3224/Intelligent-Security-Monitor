import os
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def detect_people_in_frame(frame):
    results = model(frame, verbose=False, conf=0.25)
    detected = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            label = result.names.get(cls_id, "")

            if label.lower() != "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detected.append(
                (x1, y1, x2 - x1, y2 - y1)
            )

    return detected


def main():
    video_path = os.path.join(
        "sample_videos",
        "video_01.mp4",
        "video_02.mp4"
    )

    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
    else:
        print("Video file not found.")
        return

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print("People detection started...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        people = detect_people_in_frame(frame)

        for (x, y, w, h) in people:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Person",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        count_label = f"People Count: {len(people)}"

        cv2.rectangle(
            frame,
            (15, 15),
            (230, 55),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            count_label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Intelligent Security Monitor",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Video processing completed.")


if __name__ == "__main__":
    main()