# 图像描述生成系统

本项目实现了基于深度学习的自动图像标题生成（Image Captioning），可对输入图片自动生成自然语言描述。系统采用ResNet作为图像编码器，BERT风格的Transformer作为文本解码器，支持COCO等主流数据集，具备训练、推理、批量处理等完整流程，适用于学术研究与实际应用。

---

## 主要特性

- **深度神经网络结构**：ResNet152提取图像特征，Transformer解码生成文本。
- **BERT兼容Tokenizer**：支持BERT分词器，词表与BERT一致。
- **灵活配置**：所有参数均可通过`config.json`配置。
- **批量推理**：支持目录级图片批量描述生成。
- **混合精度与多GPU训练**：提升训练效率。
- **COCO数据集支持**：内置COCO格式数据加载与处理。

---

## 目录结构

.
├── apply.py                # 推理与批量处理主程序
├── train.py                # 训练主程序
├── model.py                # 模型结构定义
├── new.py                  # 数据集与DataLoader定义
├── config.json             # 配置文件
├── checkpoints/            # 训练模型权重保存目录
├── input_images/           # 推理输入图片目录（需自行创建/填充）
├── output_captions/        # 推理输出描述目录（自动生成）
└── ...                     # 其他文件

---

## 环境依赖

- Python 3.8+
- PyTorch >= 1.10
- torchvision
- transformers
- Pillow
- pycocotools
- tqdm

安装依赖（推荐虚拟环境）：

pip install torch torchvision transformers pillow pycocotools tqdm

---

## 快速开始

### 1. 配置文件

请根据实际路径修改`config.json`中的模型权重、分词器、输入输出目录等参数。

### 2. 推理/批量生成描述

将待描述图片放入`input_images/`目录，运行：

```bash
python apply.py --config config.json
```

生成的描述将保存在`output_captions/`目录下，每张图片对应一个`.txt`文件。

### 3. 训练模型

准备COCO格式数据集，修改`train.py`中的路径配置，运行：

```bash
python train.py
```

训练过程会自动保存最优模型权重到`checkpoints/`目录。

---

## 主要配置说明（config.json）

- `input_dir`/`output_dir`：推理输入输出目录
- `model_path`：模型权重文件路径
- `tokenizer_path`：BERT分词器路径
- `model_params`：模型结构参数（如Transformer层数、隐藏维度等）
- `image_preprocessing`：图像预处理参数
- `generation`：生成参数（最大长度、束宽、温度等）
- `device`：推理设备（cuda/cpu）

---

## 训练与数据准备

- 数据集需为COCO格式（含图片与caption标注json）。
- 数据加载与增强见`new.py`，支持自定义扩展。
- 支持多GPU与混合精度训练，提升大规模训练效率。

---

## 参考/致谢

- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)
- [COCO Dataset](https://cocodataset.org/)

---

## 许可证

本项目仅供学术研究与学习使用，禁止用于商业用途。模型权重及数据集版权归原作者所有。

---

如有问题或建议，欢迎提交Issue或PR。
