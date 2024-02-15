import cv2
import face_recognition
import csv
import os
import numpy as np
from datetime import datetime
import openpyxl

video_capture = cv2.VideoCapture(0)
kritika_image = face_recognition.load_image_file("C:/Users/agraw/OneDrive/Pictures/fav/myself.jpeg")
kritika_encoding = face_recognition.face_encodings(kritika_image)[0]
ekta_image = face_recognition.load_image_file("C:/Users/agraw/OneDrive/Desktop/phone/mummy.jpg")
ekta_encoding = face_recognition.face_encodings(ekta_image)[0]
known_face_encoding = [kritika_encoding, ekta_encoding]
known_faces_names = ["kritika", "ekta"]
students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
excel_filename = "C:/Users/agraw/OneDrive/Desktop/project/csv.xlsx"

# Open the existing Excel file
workbook = openpyxl.load_workbook(excel_filename)

# Select the active sheet (you can change this based on your Excel file structure)
sheet = workbook.active

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        name = ""
        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        face_names.append(name)

        if name in students and name in known_faces_names:
            students.remove(name)
            print(students)
            current_time = now.strftime("%H-%M-%S")
            
            # Append data to the Excel file
            sheet.append([name,current_date, current_time])
            workbook.save(excel_filename)

    cv2.imshow("attendance system", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
