# Optimized for best performance

import cv2
import picamera.array
import numpy as np

# Initialize the camera
camera = picamera.PiCamera()
camera.resolution = (208, 208)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create an OpenCV window
window_name = "Look at Me"
cv2.namedWindow(window_name)
cv2.resizeWindow(window_name, 1920, 1080)

# Load the placeholder image with transparency
placeholder_image = cv2.imread("image208.png", cv2.IMREAD_UNCHANGED)
placeholder_resized = cv2.resize(placeholder_image, (208, 208))
alpha = placeholder_resized[:, :, 3] / 255.0

# Create a blank canvas of size 1080x1920
canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)

class Path:
    def __init__(self, points, current_index, video_x, video_y):
        self.points = points
        self.current_index = current_index
        self.video_x = video_x
        self.video_y = video_y

    def move(self):
        target_x, target_y = self.points[(self.current_index + 1) % len(self.points)]
        if self.video_x != target_x:
            self.video_x += 10 if target_x > self.video_x else -10
        if self.video_y != target_y:
            self.video_y += 10 if target_y > self.video_y else -10

        if self.video_x == target_x and self.video_y == target_y:
            self.current_index = (self.current_index + 1) % len(self.points)

paths = [
    Path([(40, 0), (40, 300), (280, 300), (40, 300)], 0, 40, 0),
    Path([(520, 0), (520, 300), (760, 300), (760, 0)], 0, 520, 0),
    Path([(1000, 0), (1000, 300), (1240, 300), (1240, 0)], 0, 1000, 0),
    Path([(1480, 0), (1480, 150), (1630, 0), (1480, 150), (1650, 320), (1480, 150), (1480, 310), (1480, 0)], 0, 1480,0),
    Path([(30, 840), (30, 580), (290, 840), (30, 580)], 0, 30, 840),
    Path([(520, 840), (520, 580), (400, 580), (650, 580), (520, 580)], 0, 520, 840),
    Path([(940, 840), (940, 580), (1090, 730), (1240, 580), (1240, 840), (1240, 580), (1090, 730), (940, 580)], 0, 940, 840),
    Path([(1700, 580), (1480, 580), (1480, 840), (1700, 840), (1480, 840), (1480, 700), (1650, 700), (1480, 700),(1480, 580)], 0, 1700, 580)
    # ... add other paths here ...
]

try:
    with picamera.array.PiRGBArray(camera) as stream:
        for _ in camera.capture_continuous(stream, format='bgr', use_video_port=True):
            frame = stream.array
            resized_frame = cv2.resize(frame, (208, 208))
            #flipped_frame = cv2.flip(resized_frame, 0)

            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_resized = cv2.resize(resized_frame[y:y+h, x:x+w], (208, 208))
                for path in paths:
                    canvas[path.video_y:path.video_y+208, path.video_x:path.video_x+208] = face_resized
            else:
                for c in range(0, 3):
                    for path in paths:
                        canvas[path.video_y:path.video_y+208, path.video_x:path.video_x+208, c] = \
                            (1-alpha) * canvas[path.video_y:path.video_y+208, path.video_x:path.video_x+208, c] + alpha * placeholder_resized[:, :, c]

            cv2.imshow(window_name, canvas)
            
            for path in paths:
                path.move()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            stream.truncate(0)

    cv2.destroyAllWindows()

finally:
    camera.close()
