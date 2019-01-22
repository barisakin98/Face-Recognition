import cv
import cv2
import sys


FACE_CASCADE = '/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml'
EYES_CASCADE = '/usr/share/opencv/haarcascades/haarcascade_eye_tree_eyeglasses.xml'


# cascPath = sys.argv[1]
# cascPath = FACE_CASCADE
faceCascade = cv2.CascadeClassifier(FACE_CASCADE)
eyesCascade = cv2.CascadeClassifier(EYES_CASCADE)

video_capture = cv2.VideoCapture(0)

while True:
    # Capturing frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    
    for (x, y, w, h) in faces:
        center =  x + w/2, y + h/2
        cv2.ellipse( frame, center, ( w/2, h/2 ), 0, 0, 360, (255, 0, 0 ), 2, 8, 0 );

        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    eyes = eyesCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    
    for (ex, ey, ew, eh) in eyes:
        #position = x + ex + ew/2, y + ey + eh/2
        position = ex + ew/2, ey + eh/2
        radius = cv.Round( (ew + eh)*0.25 )
        cv2.circle(frame, position, radius, ( 255, 0, 255 ), 3, 8, 0)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
