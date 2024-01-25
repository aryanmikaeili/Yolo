import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from model import Yolo
from dataset import VOCDataset
from loss import YoloLoss
from accelerate.utils import set_seed
import utils
class Trainer:
    def __init__(self):
        self.lr = 2e-5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 16
        self.weight_decay = 0.0005

        self.epoch = 100
        self.num_workers = 0
        self.pin_memory = True
        self.load_model = False

        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])])
        self.train_dataset = VOCDataset(data_dir='./data', split='train', transform=transform)
        self.trainloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=self.pin_memory, num_workers=self.num_workers, drop_last=False)

        self.test_dataset = VOCDataset(data_dir='./data', split='test', transform=transform)
        self.testloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=self.pin_memory, num_workers=self.num_workers)



        self.model = Yolo(split_size=7, num_boxes=2, num_classes=20).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = YoloLoss()

    def run(self):
        for epoch in range(self.epoch):
            self.test_step(epoch)
            self.train_step(epoch)


    def train_step(self, epoch):
        self.model.train()
        loop = tqdm(self.trainloader, leave=True)
        mean_loss = []
        for batch_idx, (image, label) in enumerate(loop):
            image, label = image.to(self.device), label.to(self.device)
            output = self.model(image)
            loss = self.criterion(output, label)
            mean_loss.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loop.set_postfix(loss=loss.item())
        print(f"Mean loss at {epoch} epoch is {sum(mean_loss)/len(mean_loss)}")


    @torch.no_grad()
    def test_step(self, epoch):
        self.model.eval()
        loop = tqdm(self.testloader, leave=True)
        all_pred_boxes = []
        all_true_boxes = []
        sample_idx = 0
        for batch_idx, (image, label) in enumerate(loop):
            image, label = image.to(self.device), label.to(self.device)
            batch_size = image.shape[0]
            preds = self.model(image).reshape(-1, 7, 7, 30)

            true_bboxes = utils.cellboxes_to_boxes(label)
            pred_bboxes = utils.cellboxes_to_boxes(preds)

            for idx in range(batch_size):
                pred_bboxes_nms = utils.non_max_suppression(pred_bboxes[idx])
                for box in pred_bboxes_nms:
                    all_pred_boxes.append([sample_idx] + box.tolist())
                for box in true_bboxes[idx]:
                    if box[1] > 0:
                        all_true_boxes.append([sample_idx] + box.tolist())
                sample_idx += 1
        if epoch == 99:
            a =0
        self.visualize_results(all_true_boxes, all_pred_boxes, epoch)
        return all_pred_boxes, all_true_boxes
    def visualize_results(self, preds, gt, epoch):
        for idx in range(len(self.test_dataset)):
            image, _ = self.test_dataset.__getitem__(idx)
            pred_labels = [box for box in preds if box[0] == idx]
            gt_labels = [box for box in gt if box[0] == idx]

            image = utils.draw_boxes(image, pred_labels, gt_labels)
            image.save(f"./{idx}.jpg")


if __name__ == '__main__':
    set_seed(42)
    trainer = Trainer()
    trainer.run()
