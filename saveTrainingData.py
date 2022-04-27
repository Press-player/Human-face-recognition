import face_recognition
import cv2
import os
import pickle

font = cv2.FONT_HERSHEY_SIMPLEX
Encodings = []
Names = []


image_dir = '/home/flashone/Desktop/pyPro/demoImages/known'
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

with open('train.plk','wb') as f:
    pickle.dump(Names,f)
    pickle.dump(Encodings,f)

if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()


print('pure fine')