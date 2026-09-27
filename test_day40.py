import os
import cv2
import pandas as pd


print("======================================")
print(" Intelligent Security Monitor")
print(" Step 16 - Automated Testing")
print("======================================")


# Test 1: Check sample videos
print("\n[TEST 1] Checking sample videos...")

video1 = "sample_videos/video_01.mp4"
video2 = "sample_videos/video_02.mp4"

if os.path.exists(video1):
    print("PASS: video_01.mp4 found")
else:
    print("FAIL: video_01.mp4 not found")

if os.path.exists(video2):
    print("PASS: video_02.mp4 found")
else:
    print("FAIL: video_02.mp4 not found")


# Test 2: Open video with OpenCV
print("\n[TEST 2] Testing OpenCV video opening...")

cap = cv2.VideoCapture(video1)

if cap.isOpened():
    print("PASS: Video opened successfully")
else:
    print("FAIL: Could not open video")

cap.release()


# Test 3: Check YOLO model
print("\n[TEST 3] Checking YOLO model...")

if os.path.exists("yolov8n.pt"):
    print("PASS: yolov8n.pt found")
else:
    print("FAIL: yolov8n.pt not found")


# Test 4: Check events.csv
print("\n[TEST 4] Checking events.csv...")

if os.path.exists("events.csv"):

    try:
        df = pd.read_csv("events.csv")

        required_columns = [
            "track_id",
            "event",
            "frame",
            "timestamp_s",
            "entry_time_s",
            "exit_time_s",
            "duration_s",
            "status"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if len(missing_columns) == 0:
            print("PASS: events.csv structure is correct")
        else:
            print("FAIL: Missing columns:", missing_columns)

    except Exception as e:
        print("FAIL: Could not read events.csv")
        print("Error:", e)

else:
    print("FAIL: events.csv not found")


# Test 5: Check processed video
print("\n[TEST 5] Checking processed video...")

processed_video = "outputs/processed_video.mp4"

if os.path.exists(processed_video):

    if os.path.getsize(processed_video) > 0:
        print("PASS: Processed video exists")
    else:
        print("FAIL: Processed video is empty")

else:
    print("FAIL: Processed video not found")


# Test 6: Check important project files
print("\n[TEST 6] Checking project files...")

required_files = [
    "detect_people.py",
    "track_people.py",
    "roi_tracking.py",
    "entry_exit.py",
    "dwell_time.py",
    "save_events.py",
    "analytics.py",
    "save_video.py",
    "performance_test.py",
    "app.py"
]

all_files_found = True

for file in required_files:

    if os.path.exists(file):
        print(f"PASS: {file}")
    else:
        print(f"FAIL: {file} not found")
        all_files_found = False


# Final result
print("\n======================================")

if all_files_found:
    print("All automated project checks completed.")
else:
    print("Some project files are missing.")

print("======================================")