import torch
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image

import cv2
import numpy as np
import utils
class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split = 'train', S=7, B=2, C=20, transform=None):
        csv_file = os.path.join(data_dir, '100examples.csv')
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[idx, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace('\n', '').split()
                ]
                boxes.append([class_label, x, y, width, height])
        img_path = os.path.join(self.image_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (
                width * self.S,
                height * self.S
            )
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1
        return image, label_matrix

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])])
    dataset = VOCDataset(data_dir='./data', split='train', transform = transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, )
    loader_iter = iter(loader)
    images, labels = next(loader_iter)
    utils.cellboxes_to_boxes(labels)
    a =0

    image, label_matrix = dataset.__getitem__(5)


    image = image.resize((448, 448), Image.BILINEAR)
    box_idx = torch.where(label_matrix[:, :, 20] == 1)
    W, H = image.size
    box = label_matrix[box_idx[0], box_idx[1], 21:25]
    for i in range(len(box)):

        x_cell, y_cell = box[i, 0], box[i, 1]
        width_cell, height_cell = box[i, 2], box[i, 3]

        x = (x_cell + box_idx[1][i]) / 7
        y = (y_cell + box_idx[0][i]) / 7
        width = width_cell / 7
        height = height_cell / 7

        #draw bounding box in image and save
        x = x * W
        y = y * H
        width = width * W
        height = height * H
        image = np.array(image)
        image = cv2.rectangle(image, (int(x - width / 2), int(y - height / 2)), (int(x + width / 2), int(y + height / 2)), (255, 0, 0), 2)
    image = Image.fromarray(image)
    image.save('test.png')
    a = 0
