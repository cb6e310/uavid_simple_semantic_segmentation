
train_image_dir = '/home/song/datasets/uav/train/seq1/Images'
train_mask_dir = '/home/song/datasets/uav/train_p/seq1/TrainId'
eval_image_dir = '/home/song/datasets/uav/val/seq16/Images'
eval_mask_dir = '/home/song/datasets/uav/val_p/seq16/TrainId'
checkpoint_dir = '/home/song/code/uav_ss/ckpt'  

import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask

def mask_to_tensor(mask):
    mask = np.array(mask, dtype=np.int64)
    mask = torch.from_numpy(mask)
    return mask

image_size = (2560, 1440)  # 

image_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    transforms.Resize(image_size, interpolation=Image.NEAREST),  # 使用最近邻插值
    transforms.Lambda(mask_to_tensor),
])

os.makedirs(checkpoint_dir, exist_ok=True)

train_dataset = CustomDataset(train_image_dir, train_mask_dir, image_transform=image_transform, mask_transform=mask_transform)
eval_dataset = CustomDataset(eval_image_dir, eval_mask_dir, image_transform=image_transform, mask_transform=mask_transform)

batch_size = 8  # 
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

def count_class_pixels(dataloader, num_classes=8):
    class_counts = np.zeros(num_classes)
    for _, masks in tqdm(dataloader, desc='Counting class pixels'):
        masks = masks.numpy()
        for c in range(num_classes):
            class_counts[c] += (masks == c).sum()
    return class_counts

def calculate_weights(class_counts):
    total = class_counts.sum()
    weights = total / (len(class_counts) * class_counts)
    weights = weights / weights.sum()  #
    return weights

weights = [0.00508186, 0.0031902,  0.00693815, 0.07688007, 0.00355312, 0.00662739,
 0.8038318,  0.09389741]
# class_counts = count_class_pixels(train_dataloader)
# weights = calculate_weights(class_counts)
# print(f'Class counts: {class_counts}')
print(f'Weights: {weights}')

weights = torch.tensor(weights, dtype=torch.float).cuda()

model = smp.DeepLabV3Plus('resnet34', encoder_weights='imagenet', classes=8, activation=None)

model = torch.nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train_and_evaluate(model, train_dataloader, eval_dataloader, optimizer, scheduler, criterion, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        
        for images, masks in train_loader:
            images = images.cuda()
            masks = masks.long().cuda()  #
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loader.set_postfix(loss=running_loss / (train_loader.n + 1))
        
        avg_train_loss = running_loss / len(train_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')
        
        model.eval()
        total_loss = 0.0
        total_iou = 0.0
        num_batches = len(eval_dataloader)
        class_correct = np.zeros(8)
        class_total = np.zeros(8)
        
        eval_loader = tqdm(eval_dataloader, desc='Evaluating', unit='batch')
        with torch.no_grad():
            for images, masks in eval_loader:
                images = images.cuda()
                masks = masks.long().cuda()
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_loss += loss.item()
                eval_loader.set_postfix(loss=total_loss / (eval_loader.n + 1))
                
                outputs = outputs.argmax(dim=1)
                intersection = (outputs & masks).float().sum((1, 2))
                union = (outputs | masks).float().sum((1, 2))
                iou = (intersection + 1e-10) / (union + 1e-10)
                total_iou += iou.mean().item()
                
                for i in range(8):  
                    class_correct[i] += ((outputs == i) & (masks == i)).sum().item()
                    class_total[i] += (masks == i).sum().item()
        
        avg_eval_loss = total_loss / num_batches
        avg_eval_iou = total_iou / num_batches
        print(f'Epoch [{epoch + 1}/{num_epochs}], Eval Loss: {avg_eval_loss:.4f}, IoU: {avg_eval_iou:.4f}')
        
        for i in range(8):
            if class_total[i] > 0:
                accuracy = 100 * class_correct[i] / class_total[i]
                print(f'Class {i} Accuracy: {accuracy:.2f}%')
            else:
                print(f'Class {i} has no samples in this epoch.')

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model saved to {checkpoint_path}')

model = model.cuda()
train_and_evaluate(model, train_dataloader, eval_dataloader, optimizer, scheduler, criterion)
