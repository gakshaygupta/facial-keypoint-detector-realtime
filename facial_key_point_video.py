import cv2
import numpy as np
from utils.inference import detect_faces
from utils.inference import apply_offsets
from utils.inference import load_detection_model
import models
def face_pt_plotter(img,pt,x1,y1,offset,scale):
    imgd=img.copy()
    x=[pt[2*i]*scale[0]+x1 for i in range(15)]
    y=[pt[2*i-1]*scale[1]+y1 for i in range(1,15+1)]
    pts=zip(x,y)
    for i in pts:
        imgd[int(i[1]):int(i[1])+offset,int(i[0]):int(i[0])+offset]=255
    if all(imgd==img):
        print("same")
    return imgd
# parameters for loading data and images
detection_model_path = 'haarcascade_frontalface_default.xml'
keypoint_model_path = 'checkpoint_xception_mini.h5'
keypoint_detector=models.mini_XCEPTION(input_shape = (96, 96, 1),num_classes=30)
keypoint_detector.load_weights(keypoint_model_path)


# loading models
face_detection = load_detection_model(detection_model_path)
#keypoint_detector.summary()
# getting input model shapes for inference
print("input shape:-", keypoint_detector.input_shape)
keypoint_target_size = keypoint_detector.input_shape[1:3]
print(keypoint_target_size)


# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)#r"F:\movies\hollywood\videoplayback_3.mp4")
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)
#    plt.imshow(gray_image)
#    plt.show()
    count=0
    for face_coordinates in faces:
        
        x1, x2, y1, y2 =apply_offsets(face_coordinates, (20,40))
        gray_face = gray_image[y1:y2, x1:x2]
        scale=np.array([x2-x1,y2-y1])/96
        try:
            gray_face = cv2.resize(gray_face,keypoint_target_size)
            
        except:
            continue

        #gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        keypoints = keypoint_detector.predict(gray_face)
#        cv2.imshow("face:-{0}".format(count),face_pt_plotter(np.resize(gray_face,[96,96]),keypoints[0],0,0,1,[1,1]))
#        count=count+1
        gray_image=face_pt_plotter(gray_image,keypoints[0],x1,y1,2,scale)
    
    cv2.imshow('window_frame', cv2.resize(gray_image,(768,570)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
