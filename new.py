# dataset_module.py
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO

class CocoDataset(Dataset):
    """
    COCO数据集加载器，支持图像和文本标注的联合加载
    功能：
    - 加载COCO格式的标注文件
    - 图像预处理和增强
    - 文本tokenization
    - 支持训练/测试集划分
    """
    
    def __init__(self, image_dir, annotation_file, tokenizer, max_length=128, split='train'):
        """
        参数：
        image_dir: COCO图像目录路径
        annotation_file: COCO标注文件路径(.json)
        tokenizer: 文本tokenizer实例
        max_length: 文本最大长度
        split: 数据集划分(train/test)
        """
        # 初始化COCO API
        self.coco = COCO(annotation_file)
        
        # 配置路径和参数
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # 定义图像预处理管道
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),      # 调整图像尺寸
            transforms.ToTensor(),               # 转换为Tensor
            transforms.Normalize(                # ImageNet标准化参数
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 获取图像ID列表
        self.image_ids = list(self.coco.imgs.keys())
        
        # 测试集模式：限制样本数量
        if self.split == 'test':
            self.image_ids = self.image_ids[:100]

    def __len__(self):
        """返回数据集数量"""
        return len(self.image_ids)

    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取图像元数据
        image_id = self.image_ids[idx]
        img_meta = self.coco.imgs[image_id]
        
        # 加载图像
        image_path = f"{self.image_dir}/{img_meta['file_name']}"
        image = Image.open(image_path).convert("RGB")
        
        # 图像预处理
        image = self.transform(image)
        
        # 加载标注文本
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        caption = annotations[0]['caption']  # 取第一个标注
        
        # 文本tokenization
        text_input = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return {
            'image': image,
            'input_ids': text_input['input_ids'].squeeze(0),
            'attention_mask': text_input['attention_mask'].squeeze(0),
            'image_path': image_path,
            'caption': caption
        }

def coco_collate_fn(batch):
    """
    自定义批处理函数，用于DataLoader
    功能：
    - 堆叠图像张量
    - 对齐文本序列
    - 打包元数据
    """
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_masks': torch.stack([item['attention_mask'] for item in batch]),
        'image_paths': [item['image_path'] for item in batch],
        'captions': [item['caption'] for item in batch]
    }

def get_dataloader(dataset, batch_size=64, num_workers=4, shuffle=True):
    """
    创建DataLoader的工厂函数
    参数：
    dataset: 初始化好的数据集实例
    batch_size: 批大小
    num_workers: 数据加载线程数
    shuffle: 是否打乱数据
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=coco_collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )

# 使用示例 --------------------------------------------------
if __name__ == '__main__':
    # 初始化组件
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # 创建数据集实例
    train_dataset = CocoDataset(
        image_dir='E:\\ANNprogram\\data\\coco\\Images\\train2017',
        annotation_file='E:\\ANNprogram\\data\\coco\\Annotations\\captions_train2017.json',
        tokenizer=tokenizer,
        split='train'
    )
    
    test_dataset = CocoDataset(
        image_dir='E:\\ANNprogram\\data\\coco\\Images\\train2017',
        annotation_file='E:\\ANNprogram\\data\\coco\\Annotations\\captions_train2017.json',
        tokenizer=tokenizer,
        split='test'
    )
    
    # 创建数据加载器
    train_loader = get_dataloader(train_dataset, batch_size=32, num_workers=4)
    test_loader = get_dataloader(test_dataset, batch_size=16, shuffle=False)
    
    # # 验证数据加载
    # sample_batch = next(iter(train_loader))
    # print("Batch结构：")
    # print(f"图像张量形状: {sample_batch['images'].shape}")
    # print(f"文本输入形状: {sample_batch['input_ids'].shape}")
    # print(sample_batch['input_ids'])
    # print(f"示例标题: {sample_batch['captions'][0]}")
