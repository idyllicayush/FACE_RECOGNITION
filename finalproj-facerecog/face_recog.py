import face_recognition
import numpy as np
import cv2
import csv
from datetime import datetime

# Initialize video capture
video_capture = cv2.VideoCapture(0)  # 0 denotes the camera used to capture face(in our case it's basic first one)

# Load known faces and encodings
ayush_image = face_recognition.load_image_file("faces/ayush.jpg")
ayush_encoding = face_recognition.face_encodings(ayush_image)[0] 

Mummy_image = face_recognition.load_image_file("faces/Mummy.jpg")
Mummy_encoding = face_recognition.face_encodings(Mummy_image)[0]

Papa_image = face_recognition.load_image_file("faces/papa.jpg")
Papa_encoding = face_recognition.face_encodings(Papa_image)[0]

yashi_image = face_recognition.load_image_file("faces/yashi.jpg")
yashi_encoding = face_recognition.face_encodings(yashi_image)[0]

Angel_image = face_recognition.load_image_file("faces/Angel.jpg")
Angel_encoding = face_recognition.face_encodings(Angel_image)[0]

Anshu_image = face_recognition.load_image_file("faces/Anshu.jpg")
Anshu_encoding = face_recognition.face_encodings(Anshu_image)[0]

known_face_encodings = [ayush_encoding,Mummy_encoding, Papa_encoding, yashi_encoding , Angel_encoding, Anshu_encoding]
known_face_names = ["Ayush","Mummy","Papa","Yashi","Angel","Anshu"]

# List of students to track attendance
students = known_face_names.copy()

# Initialize variables
face_locations = []
face_encodings = []

# Get the current date and sort it in Year month and day format
current_date = datetime.now().strftime("%Y-%m-%d")

# Open CSV file for attendance logging
with open(f"{current_date}.csv", "w+", newline="") as f:
    lnwriter = csv.writer(f)

    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize the frame to 1/4 size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Recognize faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)
            
            name = "Unknown"  # Default name if no match found
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            # Display the name if the person is recognized
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
                
                # Log the attendance
                if name in students:
                    students.remove(name)
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])
        
        # Display the resulting frame
        cv2.imshow("Attendance", frame)
        
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exit key pressed.")
            break

# Release the video capture object and close the display window
video_capture.release()
cv2.destroyAllWindows()
