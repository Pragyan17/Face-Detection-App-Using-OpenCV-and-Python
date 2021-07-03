'''Steps:
1. Grab a cra-load of faces
2. Make them all black and white
3. Train the algorithm to detect faces'''


import cv2
from random import randrange  # To generate random colors


# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# For single image
# Choose an image to detect faces in
#img = cv2.imread('RDJ.jpeg')
#img = cv2.imread('ney_lm10_cr7.jpeg')

# For capturing live video from webcam
# We can give any video file if we want inside the parenthesis
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # Detects objects of different sizes of input image
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (randrange(256), randrange(256), randrange(256)), 10)

    # Live Webcam Feed
    cv2.imshow('Clever Programmer Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if 'Q' or 'q' is pressed
    if key == 81 or key == 113:
        break

# Release the videocapture object
webcam.release()

print("Code completed")

'''
#For Single Image
# Detect faces
# Detects objects of different sizes of input image
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)
# This will give coordinates of rectangle surrounding the face

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256),
                  randrange(256), randrange(256)), 10)
#cv2.rectangle(img, (455, 98), (455+501, 98+501), (0, 255, 0), 10)


# Display the image with the faces
cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey()

print("Code completed")
'''
