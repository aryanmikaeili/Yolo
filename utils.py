import torch
from torchvision.ops import batched_nms
from PIL import Image
import numpy as np
import cv2
def intersection_over_union(boxes_preds, boxes_labels, box_format = 'midpoint'):

    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    elif box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    else:
        raise ValueError('Unknown box format.')

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection

    return intersection / (union + 1e-6)

def convert_cellboxes(boxes):
    #boxes: torch tensor (B, S, S, 30)
    bboxes1 = boxes[..., 21:25]
    bboxes2 = boxes[..., 26:30]
    scores = torch.cat((boxes[..., 20].unsqueeze(0), boxes[..., 25].unsqueeze(0)), dim = 0)
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    x_cell, y_cell = best_boxes[..., 0:1], best_boxes[..., 1:2]
    width_cell, height_cell = best_boxes[..., 2:3], best_boxes[..., 3:4]
    
    cell_indices = torch.arange(7).repeat(7, 1).unsqueeze(-1).to(boxes.device)
    
    x = (x_cell + cell_indices) / 7
    y = (y_cell + cell_indices.permute(1, 0, 2)) / 7
    
    width = width_cell / 7
    height = height_cell / 7
    
    converted_boxes = torch.cat((x, y, width, height), dim = -1)
    predicted_class = boxes[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(boxes[..., 20], boxes[..., 25]).unsqueeze(-1)
    
    converted_pred = torch.cat((predicted_class, best_confidence, converted_boxes), dim = -1)
    
    return converted_pred

def cellboxes_to_boxes(boxes, S = 7):
    #boxes: torch tensor (B, S, S, 30)
    converted_pred = convert_cellboxes(boxes)
    
    B = converted_pred.size(0)
    
    all_bboxes = converted_pred.reshape(B, S * S, -1)

    return all_bboxes
    a = 0

def non_max_suppression(bboxes, iou_threshold = 0.5, confidence_threshold = 0.4):
    #bboxes: (N, 6)
    #boxes: (N, 6) -> (N2, 4)

    #prune boxes that have confidence < threshold
    bboxes = bboxes[bboxes[:, 1] > confidence_threshold]
    bboxes_coords = bboxes[:, 2:6]
    scores = bboxes[:, 1]
    class_bboxes = bboxes[:, 0]

    # convert xywh to x1y1x2y2
    x1 = bboxes_coords[:, 0] - bboxes_coords[:, 2] / 2
    y1 = bboxes_coords[:, 1] - bboxes_coords[:, 3] / 2
    x2 = bboxes_coords[:, 0] + bboxes_coords[:, 2] / 2
    y2 = bboxes_coords[:, 1] + bboxes_coords[:, 3] / 2

    bboxes_coords = torch.stack((x1, y1, x2, y2)).T

    keep = batched_nms(bboxes_coords, scores, class_bboxes, iou_threshold=iou_threshold)

    return bboxes[keep]


def tensor2pil_list(images):
    #images: torch tensor of shape (B, 3, H, W) in range(-1, 1)

    res = []
    for i in range(images.shape[0]):
        current = images[i].permute(1, 2, 0).cpu().detach().numpy()
        current = (current + 1) / 2 * 255
        current = current.astype(np.uint8)
        res.append(Image.fromarray(current))

    return res

def visualize_images(images, row_size):
    #images: list of PIL images
    #row_size: number of images per row
    #returns: PIL image

    num_images = len(images)
    col_size = num_images // row_size
    if num_images % row_size != 0:
        col_size += 1

    image_size = images[0].size[0]
    canvas = Image.new('RGB', (image_size * row_size, image_size * col_size))
    for i, image in enumerate(images):
        canvas.paste(image, (image_size * (i % row_size), image_size * (i // row_size)))
    return canvas

def draw_boxes(image, gt_boxes, pred_boxes):
    image = ((image.permute(1, 2, 0).cpu().detach().numpy() + 1) * 127.5).astype('uint8')
    image_pred = image.copy()
    image_gt = image.copy()
    for box in gt_boxes:
        x, y, w, h = box[3:]
        x1 = int((x - w / 2) * 448)
        y1 = int((y - h / 2) * 448)
        x2 = int((x + w / 2) * 448)
        y2 = int((y + h / 2) * 448)

        image_gt = cv2.rectangle(image_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for box in pred_boxes:
        x, y, w, h = box[3:]
        x1 = int((x - w / 2) * 448)
        y1 = int((y - h / 2) * 448)
        x2 = int((x + w / 2) * 448)
        y2 = int((y + h / 2) * 448)

        image_pred = cv2.rectangle(image_pred, (x1, y1), (x2, y2), (0, 255, 0), 2)

    image_gt = Image.fromarray(image_gt)
    image_pred = Image.fromarray(image_pred)

    image = visualize_images([image_gt, image_pred], 2)

    return image

