import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John',
          'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = r'D:\Computer Vision Courses\OpenCv free code camp\projects\face-detection-opencv-builtIn'

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# p=[]
# for i in os.listdir(r'D:\Computer Vision Courses\OpenCv free code camp\projects\face-detection-opencv-builtIn'):
#     p.append(i)
# print(p)

features = []
labels = []


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)\



create_train()

print("Training done!!!")

print(f'Length of the features={len(features)}')
print(f'Length of the labels={len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on the features ist and the labels list

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')  # trained file save to use later
np.save('features.npy', features)
np.save('labels.npy', labels)
