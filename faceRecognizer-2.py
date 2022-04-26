import face_recognition
import cv2
import os

font = cv2.FONT_HERSHEY_SIMPLEX
Encodings = []
Names = []
# images to traing path
image_dir = 'training images path'
# extraction of raw information  of each image 
for root, dir, files in os.walk(image_dir):
    print(files)
    for file in files:
        path = os.path.join(root,file)
        print('path: ', path)
        name = os.path.splitext(file)[0]
        print ('name: ', name)
        person = face_recognition.load_image_file(path)
        personEncoding = face_recognition.face_encodings(person)[0]
        Encodings.append(personEncoding)
        Names.append(name)

# load image to test the recognizer

image_dir = 'test images path'

for root, dir, files in os.walk(image_dir):
    print(files)
    for file in files:
        testImagePath = os.path.join(root,file)
        testImage = face_recognition.load_image_file(testImagePath)
        face_locations = face_recognition.face_locations(testImage)
        allEncodings = face_recognition.face_encodings(testImage, face_locations)
        testImage = cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)
# once images are loaded the next "for" loop through the allEncodings array to find matches
        for (top,right,bottom,left), face_encoding in zip(face_locations,allEncodings): 
            name = 'unknown person'
            matches = face_recognition.compare_faces(Encodings, face_encoding)
            if True in matches: 
                first_match_index = matches.index(True)
                name = Names[first_match_index]
            cv2.rectangle(testImage,(left,top),(right,bottom),(0,0,255),2)
            cv2.putText(testImage,name, (left,top-6),font,.75,(255,0,255),1)

# show every test image
        cv2.imshow('myWindow',testImage)
        cv2.moveWindow('myWindow',0,0)

        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()


print('pure fine')