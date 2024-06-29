from retinaface import RetinafaceDetector
from retinaface.align_faces import warp_and_crop_face, get_reference_facial_points

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
from scipy import ndimage

def process(img, output_size):
    _, facial5points = detector(img)
    # facial5points = np.reshape(facial5points[0], (2, 5))

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)
    
    for idx_row in range(facial5points.shape[0]):
        temp_facial5points = np.reshape(facial5points[idx_row], (2, 5))
        dst_img = warp_and_crop_face(img, temp_facial5points, reference_pts=reference_5pts, crop_size=output_size)
        cv.imwrite('{}_retinaface_aligned_{}x{}.png'.format(idx_row, output_size[0], output_size[1]), dst_img)

def alignment_procedure(img, left_eye, right_eye, idx_row):
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
        angle = np.arccos(cos_a)
        angle = (angle*180)/math.pi

        if direction==-1:
            angle = 90 - angle

        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
        # img = Image.fromarray(img)
        # img = np.array(img.rotate(direction * angle, fillcolor='#000'))
        rotated = ndimage.rotate(img, angle)
        face_size = 160
        face = cv.resize(rotated,(face_size, face_size), interpolation=cv.INTER_AREA)
        cv.imwrite('rotate_img_{}.png'.format(idx_row), face)
        return face
    
    return img

### Mobinet backbone 
detector  = RetinafaceDetector(net='mnet', type='cpu').detect_faces
img  = cv.imread('/home/minhthanh/hinh-nen-girl-dep-viet-nam-21.jpg')
bounding_boxes, landmarks = detector(img)

for idx_row in range(bounding_boxes.shape[0]):
    img2 = img[int(bounding_boxes[idx_row,1]):int(bounding_boxes[idx_row,3]), int(bounding_boxes[idx_row,0]):int(bounding_boxes[idx_row,2])]
    left_eye = ((int(landmarks[idx_row,0]), int(landmarks[idx_row, 5])))
    right_eye = ((int(landmarks[idx_row,1]), int(landmarks[idx_row, 6])))
    print('left', left_eye)
    print('right', right_eye)
    rotated_image = alignment_procedure(img2, left_eye, right_eye, idx_row)
    cv.imwrite('img_{}.png'.format(idx_row), img2)
    cv.imshow('img', rotated_image)
    cv.waitKey(0)

### Resnet backbone
# detector  = RetinafaceDetector(net='mnet').detect_faces
# img  = cv.imread('./imgs/DSC_8221.jpg')
# bounding_boxes, landmarks = detector(img)
# print(bounding_boxes)
