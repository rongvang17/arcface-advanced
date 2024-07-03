import cv2
import torch
import numpy as np
import time
import pandas as pd
import threading
import math
import os
import multiprocessing
import queue
import argparse
import warnings

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization # ignore
from torchvision import transforms
from retinaface import RetinafaceDetector
from retinaface.align_faces import warp_and_crop_face, get_reference_facial_points
from arcface import ArcFace # ignore
from tqdm import tqdm
import torchvision.transforms as transforms
from recognition_model import Backbone
from datetime import datetime
from AgeNet.models import Model

warnings.simplefilter("ignore") # ignore all warnings


frame_size = (640,640)
IMG_PATH = './data/test_images'
DATA_PATH = './data'

# convert image to tensor
def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor()
            # fixed_image_standardization
        ])
    return transform(img)

# load faces_list
def load_faceslist(embeddings_path, names_path):
    # read embeddings
    embeddings_df = pd.read_csv('embeddings.csv')
    embeddings_np = embeddings_df.to_numpy()
    embeddings = torch.tensor(embeddings_np, dtype=torch.float)

    # read names
    names_df = pd.read_csv('names.csv')
    names_np = names_df['Name'].to_numpy()
    names = names_np.tolist()
    # print(names)

    return embeddings, names

# check face matching
def inference(face, embeddings, recognition_model):
    # embeds = []

    # old
    # with torch.no_grad():
    #     emb1 = face_rec.calc_emb(face)

    pil_img = Image.fromarray(face)
    # new
    with torch.no_grad():
        a = recognition_model(trans(pil_img).unsqueeze(0))
        emb1 = a[0].numpy()
        
    cosin_arr = []
    word1_embedding = emb1.reshape(1, -1)

    for i in range(embeddings.size(0)):
        word2_embedding = embeddings[i].detach().numpy().reshape(1, -1)
        similarity = cosine_similarity(word1_embedding, word2_embedding)[0][0]
        cosin_arr.append(similarity)

    index_max_cosin = cosin_arr.index(max(cosin_arr))
    if cosin_arr[index_max_cosin] > 0.6:      # matching
        return index_max_cosin, cosin_arr[index_max_cosin]
    else:
        return -1, 0
    
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

# task 1: save all images
def save_faces_detection(save_img_path, frame):
    """save all images"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(save_img_path, 'img_{}.jpg'.format(timestamp)), frame)

# task 2: recognition frame
def recognition_faces_detection(save_detection_img_path, frame):
    """recognition faces"""
    global detector, recognition_model, embeddings, names
    
    # print("prepare recognition processing......")
    pro_frame = cv2.resize(frame, (480, 480), interpolation=cv2.INTER_AREA)
    bounding_boxes, landmarks = detector(pro_frame)
    if bounding_boxes is not None:
        for idx_row in range(bounding_boxes.shape[0]):
            box_img = (int(bounding_boxes[idx_row,0]), int(bounding_boxes[idx_row,1]),
                        int(bounding_boxes[idx_row,2]), int(bounding_boxes[idx_row,3]))
            check_shape = (box_img[2] - box_img[0]) >=100 and (box_img[3] - box_img[1]) >= 100
            # print(check_shape)
            if check_shape:
                left_eye = ((int(landmarks[idx_row,0]), int(landmarks[idx_row, 5])))
                right_eye = ((int(landmarks[idx_row,1]), int(landmarks[idx_row, 6])))
                rotated_image = alignment_procedure(pro_frame, left_eye, right_eye, box_img)
                face_recognition = extract_face(rotated_image)
                gender, age = face_attributes(rotated_image)
                if face_recognition is not None:
                    idx, score = inference(face_recognition, embeddings, recognition_model)
                    print('idx;', idx, 'score:', score)
                    if idx != -1:
                        print('name face: {} - gender: {} - age: {}'.format(names[idx], gender, age))
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(os.path.join(save_detection_img_path, '{}_{}_{}_{}.jpg'.format(names[idx], gender, age, timestamp)), face_recognition)
                        # pro_frame = cv2.rectangle(pro_frame, (int(bounding_boxes[idx_row,0]), int(bounding_boxes[idx_row,1])), 
                        #                     (int(bounding_boxes[idx_row,2]), int(bounding_boxes[idx_row,3])), (0,0,255), 6)
                        # pro_frame = cv2.putText(pro_frame, '{:.2f}'.format(score) + names[idx],
                        #                     (int(bounding_boxes[idx_row,0]), int(bounding_boxes[idx_row,1])),
                        #                     cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
                    else:
                        pass
                        # pro_frame = cv2.rectangle(pro_frame, (int(bounding_boxes[idx_row,0]), int(bounding_boxes[idx_row,1])), 
                        #                     (int(bounding_boxes[idx_row,2]), int(bounding_boxes[idx_row,3])), (0,0,255), 6)
                        # pro_frame = cv2.putText(pro_frame,'Unknown', (int(bounding_boxes[idx_row,0]), int(bounding_boxes[idx_row,1])), 
                        #                     cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
                    

def transform(image):
        return transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
            ]
        )(image)


def face_attributes(rotated_image):
    """determines face attributes such as gender, age, emotions"""
    global face_attributes_model

    face = cv2.resize(rotated_image,(64, 64), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    img_pil = Image.fromarray(face)
    img = transform(img_pil)

    with torch.no_grad():
        genders, ages = face_attributes_model(img)
        print(genders, ages)

    genders = torch.round(genders)
    ages = torch.round(ages).item()
    label = "Man" if genders == 0 else "Woman"

    return label, ages


def main():
    parser = argparse.ArgumentParser(description='save detection images')
    parser.add_argument('--save_img_path', help='save all img')
    parser.add_argument('--save_detection_img_path', help='save all recognition img')

    args = parser.parse_args()
    SAVE_IMG_PATH = args.save_img_path
    SAVE_DETECTION_IMG_PATH = args.save_detection_img_path

    # checking device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # detection module
    global detector
    detector  = RetinafaceDetector(net='mnet', type='cpu').detect_faces

    # recognition module
    global recognition_model
    recognition_model = Backbone(num_layers=50, drop_ratio=0.5, mode='ir_se')
    recognition_weights = torch.load("pretrained/model_ir_se50.pth", 
                         map_location=torch.device('cpu'))
    recognition_model.load_state_dict(recognition_weights)
    recognition_model.cpu()
    recognition_model.eval()

    # faces attributes
    global face_attributes_model
    face_attributes_model = Model()
    face_attributes_model.load_state_dict(torch.load("/home/minhthanh/directory_env/my_env/arcFace-retinaFace/pretrained/faces_attributes.pt",
                                     map_location=torch.device("cpu")))
    face_attributes_model.cpu()
    face_attributes_model.eval()

    # set frame size
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)

    # load faces_list
    embeddings_path = 'embeddings.csv'
    names_path = 'names.csv'
    global embeddings, names
    embeddings, names = load_faceslist(embeddings_path, names_path)

    # inference
    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            thread_1 = threading.Thread(target=save_faces_detection,
                                        args=(SAVE_IMG_PATH, frame))
            thread_2 = threading.Thread(target=recognition_faces_detection,
                                        args=(SAVE_DETECTION_IMG_PATH, frame))

            thread_1.start()
            thread_2.start()

            thread_1.join()
            thread_2.join()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
