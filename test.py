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

# 定义数据集类
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

# 自定义的mask转换函数
def mask_to_tensor(mask):
    mask = np.array(mask, dtype=np.int64)
    mask = torch.from_numpy(mask)
    return mask

# 数据增强和转换
image_transform = transforms.Compose([
    transforms.ToTensor(),
])

# 路径配置
os.makedirs(output_dir, exist_ok=True)

# 创建数据集
eval_dataset = CustomDataset(eval_image_dir, image_transform=image_transform)

# 数据加载器
batch_size = 1  # 由于图像较大，batch_size可以适当调整
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# 加载模型
model = smp.DeepLabV3Plus('resnet34', encoder_weights=None, classes=8, activation=None)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(checkpoint_dir))
model = model.cuda()
model.eval()

# 预测并保存标签
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
