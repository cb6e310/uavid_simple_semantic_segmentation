eval_image_dir = '/home/song/datasets/uav/val/seq16/Images'
eval_mask_dir = '/home/song/datasets/uav/val_p/seq16/TrainId'
checkpoint_dir = '/home/song/code/uav_ss/ckpt/model_epoch_10.pth'  
output_dir = '/home/song/code/uav_ss/eval_outputs'  
import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir=None, image_transform=None):
        self.image_dir = image_dir
        self.image_transform = image_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        if self.image_transform:
            image = self.image_transform(image)
        
        return image, img_path

def mask_to_tensor(mask):
    mask = np.array(mask, dtype=np.int64)
    mask = torch.from_numpy(mask)
    return mask

image_transform = transforms.Compose([
    transforms.ToTensor(),
])

os.makedirs(output_dir, exist_ok=True)

eval_dataset = CustomDataset(eval_image_dir, image_transform=image_transform)

batch_size = 1  
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

model = smp.DeepLabV3Plus('resnet34', encoder_weights=None, classes=8, activation=None)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(checkpoint_dir))
model = model.cuda()
model.eval()

with torch.no_grad():
    for images, img_paths in tqdm(eval_dataloader, desc='Predicting'):
        images = images.cuda()
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()

        for pred, img_path in zip(preds, img_paths):
            pred_image = Image.fromarray(pred.astype(np.uint8), mode='L')
            img_name = os.path.basename(img_path)
            pred_image.save(os.path.join(output_dir, img_name))

print(f'Prediction labels saved to {output_dir}')
