import torch
from torchvision import transforms
from PIL import Image
import json
from model import CNNModel,ImageCaptioningModel,TextDecoder
from pathlib import Path
import shutil
from typing import Dict, Any
from transformers import BertTokenizer

class ImageCaptioningApp:
    def __init__(self, config_path: str = "config.json"):
        # 加载配置文件
        self.config = self._load_config(config_path)
        
        # 设备设置
        self.device = torch.device(self.config.get("device", 
            "cuda" if torch.cuda.is_available() else "cpu"))
        
        # 初始化组件
        self._init_tokenizer()
        self._init_model()
        self._init_transforms()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载并验证配置文件"""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"配置文件 {config_path} 未找到")
        except json.JSONDecodeError:
            raise RuntimeError("配置文件格式错误")

        # 设置默认值
        defaults = {
            "model_params": {
                "latent_dim": 512,
                "nhead": 8,
                "num_layers": 6,
                "dim_feedforward": 2048
            },
            "generation": {
                "max_length": 30,
                "beam_size": 5,
                "temperature": 1.0
            }
        }
        
        # 合并配置
        return self._deep_update(defaults, config)

    def _deep_update(self, base: Dict, update: Dict) -> Dict:
        """深度合并字典"""
        for key, val in update.items():
            if isinstance(val, dict):
                base[key] = self._deep_update(base.get(key, {}), val)
            else:
                base[key] = val
        return base

    def _init_tokenizer(self):
        """初始化tokenizer"""
        tokenizer_path = self.config.get("tokenizer_path", "bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    def _init_model(self):
        """初始化模型结构"""
        # 编码器
        encoder = CNNModel(
            embedding_size=self.config["model_params"]["latent_dim"]
        )
        
        # 解码器
        decoder = TextDecoder(
            latent_dim=self.config["model_params"]["latent_dim"],
            vocab_size=self.tokenizer.vocab_size,
            nhead=self.config["model_params"]["nhead"],
            num_layers=self.config["model_params"]["num_layers"],
            dim_feedforward=self.config["model_params"]["dim_feedforward"],
            max_seq_length=self.config["model_params"].get("max_seq_length", 128)
        )
        
        # 完整模型
        self.model = ImageCaptioningModel(encoder, decoder).to(self.device)
        
        # 加载权重
        model_path = Path(self.config["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _init_transforms(self):
        """初始化图像预处理流水线"""
        img_config = self.config.get("image_preprocessing", {})
        self.transform = transforms.Compose([
            transforms.Resize(img_config.get("resize", (224, 224))),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=img_config.get("mean", [0.485, 0.456, 0.406]),
                std=img_config.get("std", [0.229, 0.224, 0.225])
            )
        ])

    def generate_caption(self, image_path: str, **kwargs) -> str:
        """生成图像描述（支持参数覆盖）"""
        # 合并配置参数与临时参数
        gen_params = self.config["generation"].copy()
        gen_params.update(kwargs)
        
        # 处理图像
        image_tensor = self._process_image(image_path)
        
        # 编码图像
        with torch.no_grad():
            encoded_vec = self.model.encoder(image_tensor)
        
        # 生成文本
        generated_ids = self.model.decoder.generate(
            encoded_vec,
            start_token_id=self.tokenizer.cls_token_id,
            end_token_id=self.tokenizer.sep_token_id,
            **gen_params
        )
        
        # 解码文本
        return self._postprocess_text(generated_ids)

    def _process_image(self, image_path: str) -> torch.Tensor:
        """处理输入图像"""
        try:
            image = Image.open(image_path).convert("RGB")
        except IOError:
            raise ValueError("不支持的图像格式")
        
        return self.transform(image).unsqueeze(0).to(self.device)

    def _postprocess_text(self, token_ids: list) -> str:
        """后处理生成文本"""
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return text.capitalize()
    def process_batch(self):
        """批量处理目录中的图片"""
        input_dir = Path(self.config.get("input_dir", "input"))
        output_dir = Path(self.config.get("output_dir", "output"))
    
        # 创建输出目录（包含父目录）
        output_dir.mkdir(parents=True, exist_ok=True)
    
        # 支持的图片格式
        img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    
        # 遍历输入目录
        for img_path in input_dir.glob("*"):
            if img_path.suffix.lower() in img_exts:
                try:
                    # 生成描述
                    caption = self.generate_caption(str(img_path))
                
                    # 构建输出路径
                    output_path = output_dir / f"{img_path.stem}.txt"
                
                    # 保存结果
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(caption)
                    
                    print(f"成功处理: {img_path.name}")
                
                except Exception as e:
                    print(f"处理失败 {img_path.name}: {str(e)}")
                    # 可选：将错误图片移动到错误目录
                    error_dir = output_dir / "errors"
                    error_dir.mkdir(exist_ok=True)
                    shutil.copy(img_path, error_dir / img_path.name)\

if __name__ == "__main__":
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description="图像描述生成工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default="config.json",
                        help="配置文件路径")
    args = parser.parse_args()
    
    # 初始化应用
    app = ImageCaptioningApp(args.config)
    
    # 执行批量处理
    print("\n" + "="*40)
    print(f"输入目录: {app.config['input_dir']}")
    print(f"输出目录: {app.config['output_dir']}")
    print("="*40 + "\n")
    
    app.process_batch()
    
    print("\n处理完成！")

# 示例配置文件 config.json
"""
{
    "model_path": "models/best_model.pth",
    "tokenizer_path": "bert-base-uncased",
    
    "model_params": {
        "latent_dim": 768,
        "nhead": 12,
        "num_layers": 6,
        "dim_feedforward": 3072,
        "max_seq_length": 256
    },
    
    "image_preprocessing": {
        "resize": [224, 224],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    
    "generation": {
        "max_length": 30,
        "beam_size": 5,
        "temperature": 0.9
    },
    
    "device": "cuda"
}
"""
