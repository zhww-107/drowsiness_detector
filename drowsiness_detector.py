from imutils import face_utils
from scipy.spatial import distance
import cv2
import dlib
import imutils
import pygame
import time

# Initializing the alert sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert_sound.wav")
default_volume = 0.2

# Eye-Aspect-Ratio data
EAR_threshhold = 0.17	# One valid frame is counted when EAR is lower than this value
frame_count = 0			# Number of frames when EAR is lower than EAR_threshhold 
EAR_total_frame = 25	# Having frame_count larger than this value is considered drowsiness


# Play the alarm in a given volume
def alert(volume):
    alert_sound.set_volume(volume)
    alert_sound.play()

# Given an eye landmark, compute its eye_aspect_ratio
def eye_aspect_ratio(eye):
    v1 = distance.euclidean(eye[1], eye[5])
    v2 = distance.euclidean(eye[2], eye[4])
    h1 = distance.euclidean(eye[0], eye[3])
    return (v1 + v2) / (2 * h1)


# Initialize the face detector and Facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Access the camera
cap = cv2.VideoCapture(0)


# Main loop for drowsiness detection
while True:
	# Read the camera input, resize it, and concert it to grayscale frame
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in grayscale frame
    bounds = detector(raw,0)

    for bound in bounds:
    	# Predict facial landmarks for each detected face
        shape = predictor(raw,bound)
        # Convert the facial lanmarks into a 1-D numpy array (x, y)
        shape = face_utils.shape_to_np(shape)

        # Left and right eyes' indexes for facial landmarks
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # The main EAR is the average of left and right eye's EAR
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        EAR = (left_EAR + right_EAR) / 2
        
        # Draw the facial landmarks for left eye
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Draw the facial landmarks for right eye
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Alarm when drowsiness is detected
        if EAR < EAR_threshhold:
            frame_count += 1
			# Volume increases gradually
            if frame_count >= EAR_total_frame:
                alert(0.2 + (frame_count - 25) * 0.2)
                time.sleep(3)
        else:
            frame_count = 0

        # Display informations
        cv2.putText(frame, "Frame: {:.0f}".format(frame_count), (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "Eye-Aspect-Ratio: {:.2f}".format(EAR), (30, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "Press Q to exit.", (410, 320),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # Display the frame
    cv2.imshow("Drowsiness_Detector", frame)

    # Provide a way to exit the program -- pressing "Q"
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
    	break

cv2.destroyAllWindows()