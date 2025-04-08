import cv2
import face_recognition

# Load known faces and their names
known_faces = []
known_names = []

# Load sample images and encode them
image_1 = face_recognition.load_image_file("person1.jpg")
encoding_1 = face_recognition.face_encodings(image_1)[0]
known_faces.append(encoding_1)
known_names.append("Person 1")

image_2 = face_recognition.load_image_file("person2.jpg")
encoding_2 = face_recognition.face_encodings(image_2)[0]
known_faces.append(encoding_2)
known_names.append("Person 2")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert frame to RGB (face_recognition uses RGB, OpenCV uses BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # Find the best match
        if True in matches:
            matched_index = matches.index(True)
            name = known_names[matched_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display name
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
