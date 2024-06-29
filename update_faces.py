import glob
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import shutil
import argparse
import threading

from tqdm import tqdm
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image
from arcface import ArcFace
from model import Backbone

# checking device
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# convert image to tensor
def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor()
        ])
    return transform(img)

def check_name_exits(usr, ALL_IMG_PATH):
    exits = False
    for all_usr in os.listdir(ALL_IMG_PATH):
        if usr == all_usr:
            print("{} already exists in DB".format(usr))
            exits = True
            break
    return exits

# add_new_embeddings
def create_embeddings(ALL_IMG_PATH, ADD_IMG_PATH):
    embeddings = []
    names = []
    cnt_new_faces = 0
    requirement_add_infor = False
    # face_rec = ArcFace.ArcFace() # recognition module

    model = Backbone(num_layers=50, drop_ratio=0.5, mode='ir_se')
    ckpt = torch.load("InsightFace_Pytorch/pretrained/model_ir_se50.pth", map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)
    model.cpu()
    model.eval()

    for usr in os.listdir(ADD_IMG_PATH):
        check_exist = check_name_exits(usr, ALL_IMG_PATH) #
        if not check_exist:
            temp_img_path = os.path.join(ALL_IMG_PATH, usr)
            usr_path = os.path.join(ADD_IMG_PATH, usr)
            # print(usr_path)
            shutil.copytree(usr_path, temp_img_path)

            embeds = []
            name_images = os.listdir(usr_path)
            # print(names)
            for name_image in name_images:
                requirement_add_infor = True
                image_path = os.path.join(usr_path, name_image)
                img = cv2.imread(image_path)
                img = cv2.resize(img,(112, 112), interpolation=cv2.INTER_AREA)
                # old
                # with torch.no_grad():
                #     emb1 = face_rec.calc_emb(img)
                #     embeds.append(emb1) # 1 anh, kich thuoc [1,512]

                # new
                pil_img = Image.fromarray(img)
                with torch.no_grad():
                    a = model(trans(pil_img).unsqueeze(0))
                    emb1 = a[0].numpy()
                    embeds.append(emb1) # 1 anh, kich thuoc [1,512]

            if len(embeds) == 0:
                continue
            embeds = [torch.tensor(embedd).unsqueeze(0) for embedd in embeds]
            embedding = torch.cat(embeds) 
            embeddings.append(embedding) # 1 cai list n cai [1,512]
            names.append([usr] * embedding.shape[0])
            cnt_new_faces += 1
        else:
            continue

    return embeddings, names, requirement_add_infor, cnt_new_faces

# save embeddings, names to .csv
def save_embeddings(embeddings, names, save_data_path):
    embeddings = torch.cat(embeddings) #[n,512]

    names_flat = []
    for sublist in names:
        names_flat.extend(sublist)
    names = np.array(names_flat).reshape(-1)

    # save names to .csv
    names_csv = 'names.csv'
    names_path = os.path.join(save_data_path, names_csv)
    if os.path.exists(names_path):
        names_series = pd.Series(names)
        names_df = pd.DataFrame(names_series, columns=['Name'])
        names_df.to_csv(names_path, mode='a', header=False, index=False)
    else:
        names_series = pd.Series(names)
        names_df = pd.DataFrame(names_series, columns=['Name'])
        names_df.to_csv(names_path, index=False)

    # save embedding to .csv
    embeddings_csv = 'embeddings.csv'
    embeddings_path = os.path.join(save_data_path, embeddings_csv)
    if os.path.exists(embeddings_path):
        embeddings_np = embeddings.cpu().numpy().astype(float)
        df = pd.DataFrame(embeddings_np)
        df.to_csv(embeddings_path, mode='a', header=False, index=False)
    else:
        embeddings_np = embeddings.cpu().numpy().astype(float)
        df = pd.DataFrame(embeddings_np)
        df.to_csv(embeddings_path, index=False)


def main():
    # faces_list path
    save_data_path = '/home/minhthanh/directory_env/my_env/arcFace-retinaFace/'

    parser = argparse.ArgumentParser(description='get all_infor_path')
    parser.add_argument('--all_img_path', help='all_faces_list')
    parser.add_argument('--add_img_path', help='names_need_add')

    args = parser.parse_args()

    ALL_IMG_PATH = args.all_img_path
    ADD_IMG_PATH = args.add_img_path

    embeddings, names, requirement_add_infor, cnt_new_faces = create_embeddings(ALL_IMG_PATH, ADD_IMG_PATH)
    if requirement_add_infor:
        save_embeddings(embeddings, names, save_data_path)
        print("updated {} peoples success".format(cnt_new_faces))
        print("There are a total of {} people in DB".format(len(os.listdir(ALL_IMG_PATH))))

if __name__ == '__main__':
    main()