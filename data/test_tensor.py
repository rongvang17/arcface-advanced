import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


x = torch.tensor([[1, 2, 3]])
y = torch.tensor([[4 ,5, 7], [7, 9, 8], [1, 2, 2.7]])

# x = x.expand(y.size())

word1_embedding = x.reshape(1, -1)
a = []

for i in range(y.size(0)):
    word2_embedding = y[i].reshape(1, -1)
    similarity = cosine_similarity(word1_embedding, word2_embedding)[0][0]
    a.append(similarity)
    similarity2 = cosine_similarity(word1_embedding, word2_embedding)
    print(similarity2)

index_min = a.index(max(a))

print(index_min)
# z = x*y
# # z = pow((x-y), 2)

# print(z)