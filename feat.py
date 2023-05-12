import cv2
import os
import numpy as np

# def max_pooling(img):
#     pool_size = 2

#     
#     new_h = img.shape[0] // pool_size
#     new_w = img.shape[1] // pool_size
#     new_d = img.shape[2]

#     
#     pooled_img = np.zeros((new_h, new_w, new_d))

#    
#     for i in range(new_h):
#         for j in range(new_w):
#             start_i = i * pool_size
#             start_j = j * pool_size
#             end_i = start_i + pool_size
#             end_j = start_j + pool_size
#             # pooled_img[i, j] = np.max(img[start_i:end_i, start_j:end_j], axis=(0, 1))
#             pooled_img[i, j] = np.mean(img[start_i:end_i, start_j:end_j], axis=(0, 1))

#     return pooled_img
def load_images(path):
    images = []
    labels = []
    for folder_name in os.listdir(path):
        
        if folder_name == 'person':
            label = 1
        else:
            label = 0
        folder_path = os.path.join(path, folder_name)
    
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None :
                img = img.reshape(-1)
                img = img / 255.0
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)

def load_test(path):
    images = []
    labels = []
    i = 1

    sorted_img = []

    for img_name in os.listdir(path):
        label = img_name[:-4]
        # sorted_img.append(int(label))
        sorted_img.append((label))
    # sorted_img = sorted(sorted_img)
    
    for img_num in sorted_img:
        # img_name = "img_" + str(img_num) + ".png"
        img_name = img_num + ".png"
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is not None :
            img = img.reshape(-1)
            img = img / 255.0
            images.append(img)
            label = img_name[:-4]
            labels.append(label)

    return np.array(images), np.array(labels)


def load_multi(path):
    images = []
    labels = []
    for folder_name in os.listdir(path):
        
        if folder_name == 'car':
            label = 0
        elif folder_name == 'person':
            label = 1
        elif folder_name == 'airplane' :
            label = 2
        else: 
            label = 3

        folder_path = os.path.join(path, folder_name)
    
        for img_name in os.listdir(folder_path):
            
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None :
                img = img.reshape(-1)
                img = img / 255.0
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)


