import cv2

video_path = "sample_videos/video_01.mp4"  
video_path = "sample_videos/video_02.mp4"

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully")

while True: 
    ret, frame = video.read()

    if not ret:
        break

    cv2.imshow("Intelligent Security Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

print("Video processing completed")