import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from model import CNNModel, TextDecoder, ImageCaptioningModel
from new import CocoDataset, get_dataloader
from transformers import BertTokenizer
import os
from torch.optim.lr_scheduler import LambdaLR
import time
from tqdm import tqdm

# 配置参数
CONFIG = {
    "train_image_dir": 'E:\\ANNprogram\\data\\coco\\Images\\train2017',
    "train_annotation_file": 'E:\\ANNprogram\\data\\coco\\Annotations\\captions_train2017.json',
    "test_image_dir": 'E:\\ANNprogram\\data\\coco\\Images\\test2017',
    "test_annotation_file": 'E:\\ANNprogram\\data\\coco\\Annotations\\captions_test2017.json',
    "val_image_dir": 'E:\\ANNprogram\\data\\coco\\Images\\val2017',
    "val_annotation_file": 'E:\\ANNprogram\\data\\coco\\Annotations\\captions_val2017.json',
    "vocab_size": 30522,  # 与BERT tokenizer一致
    "max_seq_length": 128,
    "embedding_size": 512,
    "batch_size": 48,
    "num_workers": 16,
    "learning_rate": {
        "encoder": 1e-5,
        "decoder": 1e-4
    },
    "num_epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "checkpoints",
    "pretrained_bert": "bert-base-uncased",
    "mixed_precision": True,  # 启用混合精度训练
    "multi_gpu": True,        # 启用多GPU支持
    "pin_memory": True,       # 启用内存锁页
    "grad_accumulation": 2,    # 梯度累积步数
    "warmup_steps": 2000,         # 学习率预热步数
    "min_lr_ratio": 0.1,          # 最小学习率比例
}

def create_warmup_scheduler(optimizer, warmup_steps, min_lr_ratio):
    """创建学习率预热调度器"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return min_lr_ratio + (1 - min_lr_ratio) * current_step / warmup_steps
        else:
            return 1.0
    return LambdaLR(optimizer, lr_lambda)

def main():
    # 初始化设备
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")

    # 初始化组件
    tokenizer = BertTokenizer.from_pretrained(CONFIG["pretrained_bert"])
    tokenizer.save_pretrained("./model")
    
    # 创建数据集和数据加载器（启用pin_memory）
    train_dataset = CocoDataset(
        image_dir=CONFIG["train_image_dir"],
        annotation_file=CONFIG["train_annotation_file"],
        tokenizer=tokenizer,
        max_length=CONFIG["max_seq_length"],
        split='train'
    )
    
    val_dataset = CocoDataset(
        image_dir=CONFIG["val_image_dir"],
        annotation_file=CONFIG["val_annotation_file"],
        tokenizer=tokenizer,
        max_length=CONFIG["max_seq_length"],
        split='test'
    )
    
    train_loader = get_dataloader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"]
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )

    # 初始化模型
    encoder = CNNModel(CONFIG["embedding_size"])
    decoder = TextDecoder(
        latent_dim=CONFIG["embedding_size"],
        vocab_size=CONFIG["vocab_size"],
        max_seq_length=CONFIG["max_seq_length"]
    )
    model = ImageCaptioningModel(encoder, decoder)
    
    # 多GPU并行
    if CONFIG["multi_gpu"] and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)
    model = model.to(device)

    # 混合精度训练
    scaler = GradScaler(enabled=CONFIG["mixed_precision"])
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=0.1  # 新增标签平滑
    )
    optimizer = optim.AdamW([
        {"params": encoder.parameters(), "lr": CONFIG["learning_rate"]["encoder"]},
        {"params": decoder.parameters(), "lr": CONFIG["learning_rate"]["decoder"]}
    ])

    # 训练循环
    best_val_loss = float('inf')
    global_step = 0
    scheduler = create_warmup_scheduler(
        optimizer,
        warmup_steps=CONFIG["warmup_steps"],
        min_lr_ratio=CONFIG["min_lr_ratio"]
    )
    for epoch in range(CONFIG["num_epochs"]):
        # 训练阶段
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")):
            # 数据转移至GPU
            images = batch['images'].to(device, non_blocking=True)
            captions = batch['input_ids'].to(device, non_blocking=True)
            if global_step < CONFIG["warmup_steps"]:
                scheduler.step()
            global_step += 1
            
            with autocast(enabled=CONFIG["mixed_precision"]):
                # 前向传播
                logits = model(images, captions[:, :-1])
                loss = criterion(
                    logits.view(-1, CONFIG["vocab_size"]),
                    captions[:, 1:].contiguous().view(-1)
                )
                loss = loss / CONFIG["grad_accumulation"]  # 梯度累积
                
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积
            if (step + 1) % CONFIG["grad_accumulation"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                images = batch['images'].to(device, non_blocking=True)
                captions = batch['input_ids'].to(device, non_blocking=True)
                
                with autocast(enabled=CONFIG["mixed_precision"]):
                    logits = model(images, captions[:, :-1])
                    loss = criterion(
                        logits.view(-1, CONFIG["vocab_size"]),
                        captions[:, 1:].contiguous().view(-1)
                    )
                val_loss += loss.item()
        
        # 计算指标
        train_loss /= len(train_loader) * CONFIG["grad_accumulation"]
        val_loss /= len(val_loader)
        
        # 显存清理
        torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # 保存检查点
        if val_loss < best_val_loss:
            timestamp = time.strftime("%Y%m%d_%H%M%S")  # 格式示例：20230622_153045
            best_val_loss = val_loss
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(),
                      os.path.join(CONFIG["checkpoint_dir"], f"best_model_{timestamp}.pth"))
            print(f"Saved best model with val loss: {val_loss:.4f}")

if __name__ == "__main__":
    # 初始化配置
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    
    # 设置CUDA优化
    torch.backends.cudnn.benchmark = True  # 启用CuDNN自动优化器
    torch.set_float32_matmul_precision('high')  # 矩阵乘法精度模式
    
    main()