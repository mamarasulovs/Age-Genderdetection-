# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import imutils
import time
import cv2
import requests
from time import sleep

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
res_x = 640
res_y = 480
#cap = cv2.VideoCapture('rtsp://admin:Pa$$123456@195.158.16.190:15554/cam/realmonitor?channel=1&subtype=0')
#print("After URL")

'''while True:

    print('About to start the Read command')
    ret, frame = cap.read()
    print('About to show frame of Video.')
    cv2.imshow("Capturing",frame)
    print('Running..')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''


camera.resolution = (640,480)
camera.framerate = 15
rawCapture = PiRGBArray(camera, size=(640,480))
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list=['0,2','4,6','8,12','15,20','25,32','38,43','48,53','60+']
gender_list = ['Hombre:', 'Mujer:']
emotions = ["Asustad@", "Molest@", "Disgustad@", "Feliz", "Neutral", "Triste", "Sorprendido"]
# allow the camera to warmup
time.sleep(0.1)
 
def connection_handler(data):
    try:
        payload = {'payload': data }
        req = requests.get('rtsp://admin:Pa$$123456@195.158.16.190:15554/cam/realmonitor?channel=1&subtype=0',params=payload)
        print (req.url) 
        return "ok"
    except Exception as e:
    	print ('e')

def locate_faces(image):
    faces = face_cascade.detectMultiScale(image, 1.1, 5)
    return faces  # list of (x, y, w, h)

def normalize_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.equalizeHist(face)
    face = cv2.resize(face, (350, 350))

    return face;

def find_faces(image):
    faces_coordinates = locate_faces(image)
    cutted_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in faces_coordinates]
    normalized_faces = [normalize_face(face) for face in cutted_faces]
    return zip(normalized_faces, faces_coordinates)



        
def initialize_caffe_model():
    print('Loading models...')
    age_net = cv2.dnn.readNetFromCaffe(
                        "models/deploy_age.prototxt", 
                        "models/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(
                        "models/deploy_gender.prototxt", 
                        "models/gender_net.caffemodel")
 
    return (age_net, gender_net)
 
def capture_loop(age_net, gender_net,emotion_model): 
    font = cv2.FONT_HERSHEY_SIMPLEX
    # capture frames from the camera
    flag =""
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        #/usr/local/share/OpenCV/haarcascades/
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        faces_coordinates = locate_faces(image)
        #normalized_faces = [normalize_face(face) for face in cutted_faces]
        #faceArray =  zip(normalized_faces, faces_coordinates)
        #normalized_faces, faces_coordinates = zip(*find_faces(image))
        faces = faces_coordinates
        print("Found "+str(len(faces))+" face(s)")
        #Draw a rectangle around every found face
        counter = 0
        cara=[]
        people =""
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
            face_img = image[y:y+h, x:x+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # Predict gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            #Predict emotions
            cutted_face = image[y:y + h, x:x + w]
            n_face = normalize_face(cutted_face)
            emotion_prediction,emotion_distance = emotion_model.predict(n_face)  # do prediction
            #Process data as output
            overlay_text = "%s, %s ,%s" % (gender, age, emotions[emotion_prediction])
            print(overlay_text)
            people = gender + age
            if flag != people:
                cara.append(people)
            flag = people
        if len(faces)!=0:
            if len(cara) != 0:
                print ('cara')
                msg = connection_handler(str(cara))
        else:
            cara = []
            flag = ""

            #cv2.putText(image, overlay_text ,(x,y), font, 1,(255,255,255),1,cv2.LINE_AA)

        cv2.imshow("Image", image) # comentar si no se quiere que salga el frame
        sleep(0.25)
        key = cv2.waitKey(1) & 0xFF
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

if __name__ == '__main__':
    age_net, gender_net = initialize_caffe_model()
    fisher_face_emotions = cv2.face.FisherFaceRecognizer_create()
    fisher_face_emotions.read('models/emotion_classifier_model.xml')
    capture_loop(age_net, gender_net,fisher_face_emotions)

    cap.release()
    cv2.destroyAllWindows()