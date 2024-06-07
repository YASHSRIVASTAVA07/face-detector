import pathlib
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread, Event

# Path to the Haar cascade file
cascade_path = pathlib.Path(cv2.__file__).parent / "data/haarcascade_frontalface_default.xml"

# Load the cascade classifier
clf = cv2.CascadeClassifier(str(cascade_path))

# Function to capture video and detect faces
def video_capture(stop_event, label):
    camera = cv2.VideoCapture(0)

    while not stop_event.is_set():
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)

        # Convert the frame to an image for display
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label with the new image
        label.imgtk = imgtk
        label.configure(image=imgtk)

        if stop_event.is_set():
            break

    camera.release()
    cv2.destroyAllWindows()

# Function to stop the video capture
def stop_capture():
    stop_event.set()
    root.destroy()

# Create a window with a close button and video display
root = tk.Tk()
root.title("Face Detection")

stop_event = Event()

# Create a label to display the video feed
video_label = tk.Label(root)
video_label.pack()

# Start the video capture in a separate thread
start_thread = Thread(target=video_capture, args=(stop_event, video_label))
start_thread.start()

# Create and pack the close button
close_button = tk.Button(root, text="Close", command=stop_capture)
close_button.pack(pady=20)

# Run the GUI main loop
root.mainloop()

# Ensure that the video capture thread is properly terminated
stop_event.set()
start_thread.join()
