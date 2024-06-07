import pathlib
import cv2

# Path to the Haar cascade file
cascade_path = pathlib.Path(cv2.__file__).parent / "data/haarcascade_frontalface_default.xml"
print(cascade_path)

# Load the cascade classifier
clf = cv2.CascadeClassifier(str(cascade_path))

# Start the video capture
camera = cv2.VideoCapture(0)

while True:
    # Read frame from the camera
    _, frame = camera.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around the faces
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Faces", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the capture and close the window
camera.release()
cv2.destroyAllWindows()
