
import cv2
video_path = '/Users/matt/CDIO_3_2024/image_detection/data/videos/test3.mp4'

vidcap = cv2.VideoCapture(video_path)

# Frame counter
count = 0

# Frame stride - take every x frames
frame_stride = 10  # Adjust this value as needed

# Read the first frame
success, image = vidcap.read()

# Loop through all frames
while success:
    # Save the frame if count is a multiple of frame_stride
    if count % frame_stride == 0:
        cv2.imwrite("frame_v3 %dp.png" % count, image)  # Save frame as JPEG file
    success, image = vidcap.read()  # Read the next frame
    print('Read a new frame: ', success)
    count += 1

# Release the video capture object
vidcap.release()
