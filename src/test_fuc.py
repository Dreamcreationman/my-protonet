from tqdm import tqdm
import torch
import numpy as np
import time
import os
import csv
#
# def find_items(root_dir):
#     retour = []
#     rots = [os.sep + 'rot000', os.sep + 'rot090', os.sep + 'rot180', os.sep + 'rot270']
#     for (root, dirs, files) in os.walk(root_dir):
#         for f in files:
#             r = root.split(os.sep)
#             lr = len(r)
#             label = r[lr - 2] + os.sep + r[lr - 1]
#             for rot in rots:
#                 retour.extend([(f, label, root, rot)])
#     print("== Dataset: Found %d items " % len(retour))
#     return retour
#
# find_items("E:/Datasets/omniglot/python/data")

# with open("E:/Datasets/mini-imagenet/train.csv", "r", encoding="utf-8") as f:
#     reader = csv.reader(f)
#     for i in reader:
#         print(i[0], i[1])

# a = np.arange(10)
# # b = np.arange(2, 11)
# #
# # c = np.argwhere(a == b).item()
# # print(c)
# b = [2, 5, 8]
# print(torch.randperm(a[b]))

a = torch.randn(1, 6).view(2, 3)
b = a.unsqueeze(1)
c = b.expand(2, 4, 3)
print(a)
print(b)
print(c)
print("============================")
a = torch.randn(1, 12).view(4, 3)
b = a.unsqueeze(0)
c = b.expand(2, 4, 3)
print(a)
print(b)
print(c)