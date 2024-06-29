import cv2
import numpy as np
import math
import torch
import os
import time

from PIL import Image
from facenet_pytorch import MTCNN
from datetime import datetime
from retinaface import RetinafaceDetector
from retinaface.align_faces import warp_and_crop_face, get_reference_facial_points

# get box face
def extract_face(img):
    if img is None or img.size==0:
        return None
    face_size = 112
    face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    return face

def alignment_procedure(frame, left_eye, right_eye, box_img):
    #this function aligns given face in img based on left and right eye coordinates
    
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock

    a_edge = math.sqrt((point_3rd[0] - left_eye[0])**2 + (point_3rd[1] - left_eye[1])**2)
    b_edge = math.sqrt((point_3rd[0] - right_eye[0])**2 + (point_3rd[1] - right_eye[1])**2)
    c_edge = math.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)

    if b_edge*a_edge != 0:
        cos_a = float(b_edge/c_edge)
        # print(cos_a)
        angle = np.arccos(cos_a)
        angle = (angle*180)/math.pi
        # print('{}_do'.format(angle))

        if direction==-1:
            angle = 90 - angle

        img = Image.fromarray(frame)
        img = np.array(img.rotate(direction * angle, fillcolor='#000'))
        img = img[box_img[1]:box_img[3], box_img[0]:box_img[2]]
        return img
    
    return frame[box_img[1]:box_img[3], box_img[0]:box_img[2]]

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMG_PATH = '/home/minhthanh/directory_env/my_env/arcFace-retinaFace/data2/test_images2'
count = 16
usr_name = input("Input all name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)
if not os.path.exists(USR_PATH):
    os.mkdir(USR_PATH)

### Mobinet backbone 
detector  = RetinafaceDetector(net='mnet', type='cpu').detect_faces
leap = 1
count = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
current_time = time.time()
duration = 5*60*60
end_time = current_time + duration
while cap.isOpened() and time.time() < end_time:
    isSuccess, frame = cap.read()
    if isSuccess and leap%2==0:
        bounding_boxes, landmarks = detector(frame)
        if bounding_boxes is not None:
            for idx_row in range(bounding_boxes.shape[0]):
                box_img = (int(bounding_boxes[idx_row,0]), int(bounding_boxes[idx_row,1]), 
                               int(bounding_boxes[idx_row,2]), int(bounding_boxes[idx_row,3]))
                # cv2.rectangle(frame, (box_img[0], box_img[1]), (box_img[2], box_img[3]), (0, 255, 0), 2)
                # cv2.imshow('crop_img', frame)
                check_shape = (box_img[2] - box_img[0]) >=100 and (box_img[3] - box_img[1]) >= 100
                if check_shape:
                    left_eye = ((int(landmarks[idx_row,0]), int(landmarks[idx_row, 5])))
                    right_eye = ((int(landmarks[idx_row,1]), int(landmarks[idx_row, 6])))
                    rotated_image = alignment_procedure(frame, left_eye, right_eye, box_img)
                    face = extract_face(rotated_image)
                    if face is not None:
                        cv2.imwrite(os.path.join(USR_PATH, '{}.png'.format(count)), face)
    leap += 1
    count += 1
    cv2.imshow('Face Capturing', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
