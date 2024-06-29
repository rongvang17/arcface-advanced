from arcface import ArcFace
face_rec = ArcFace.ArcFace()
emb1 = face_rec.calc_emb("/home/minhthanh/directory_env/my_env/arcFace-retinaFace/img_screenshot_25.06.2024.png")
print(emb1)
# emb2 = face_rec.calc_emb("~/Downloads/test2.jpg")
# face_rec.get_distance_embeddings(emb1, emb2)
