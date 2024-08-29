import cv2
import os
import numpy as np
from datetime import datetime
import face_recognition

# Load student images and create encodings
def load_student_images(directory):
    student_images = []
    student_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            student_names.append(os.path.splitext(filename)[0])
            student_images.append(image_path)
    return student_images, student_names

def encode_image(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        return encodings[0]
    else:
        return None

def mark_attendance(name):
    today = datetime.now().strftime('%Y-%m-%d')
    filename = f'Attendance_{today}.csv'
    
    # Check if the file exists and read existing records
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_records = [line.split(',')[0] for line in f.readlines()]
    else:
        existing_records = []

    # Only write if the name is not already present
    if name not in existing_records:
        with open(filename, 'a') as f:
            now = datetime.now()
            dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'{name},{dt_string}\n')
        return False
    return True

# Initialize known face encodings and their names
student_images, student_names = load_student_images('images')
student_encodings = [encode_image(image) for image in student_images]
student_encodings = [encoding for encoding in student_encodings if encoding is not None]

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Process each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the current face with known encodings
        matches = face_recognition.compare_faces(student_encodings, face_encoding)
        face_distances = face_recognition.face_distance(student_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = student_names[best_match_index]
            already_marked = mark_attendance(name)
            if already_marked:
                name = "Already marked present"
                frame_color = (0, 255, 0)  # Red color
            else:
                frame_color = (0, 255, 0)  # Green color
        else:
            name = "Cannot recognize you"
            frame_color = (0, 0, 255)  # Red color

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), frame_color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, frame_color, 2)

    # Display the resulting image
    cv2.imshow('Attendance system', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
