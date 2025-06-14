# 图像描述生成系统 - 安装包版使用说明

本说明适用于通过 Nuitka 打包并用 Inno Setup 制作的**图像描述生成系统**安装包。该系统可自动为图片生成自然语言描述，适合批量图片自动标注、辅助内容生成等场景。

---

## 安装步骤

1. **运行安装包**
   
   - 双击运行安装包（如 `ImageCaptioningSetup.exe`），根据提示完成安装。
   - 默认安装目录如：`C:\Program Files\ImageCaptioning`。

2. **准备模型和分词器**
   
   - 将训练好的模型权重（如 `best_model_xxx.pth`）和分词器（如 `bert-base-uncased` 或自定义分词器文件夹）放入安装目录下，或根据 `config.json` 配置路径放置。

3. **准备图片**
   
   - 将待描述的图片（支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`）放入 `input_images` 文件夹（可在安装目录下新建）。

4. **配置参数**
   
   - 使用文本编辑器打开安装目录下的 `config.json`，根据实际情况修改以下内容：
     - `model_path`：模型权重文件路径
     - `tokenizer_path`：分词器路径
     - `input_dir`：输入图片文件夹路径（如 `input_images`）
     - `output_dir`：输出描述文件夹路径（如 `output_captions`）
     - 其他参数可按需调整

---

## 使用方法

### 方式一：命令行运行

1. 打开命令提示符（Win+R 输入 `cmd` 回车）。
2. 切换到安装目录，例如：
   
   ```bash
   cd "C:\Program Files\ImageCaptioning"
   ```

3. 执行批量描述生成（无需 Python 环境）：
   
   ```bash
   ImageCaptioning.exe --config config.json
   ```
- 也可直接双击 `ImageCaptioning.exe`，程序会自动读取 `config.json` 并处理图片。
4. 处理完成后，描述文本将保存在 `output_dir` 指定的文件夹下，每张图片对应一个 `.txt` 文件。

### 方式二：图形界面（如有）

如安装包包含图形界面（GUI），可直接双击桌面快捷方式或开始菜单图标，按界面提示操作。

---

## 常见问题

- **模型或分词器未找到**  
  请确保 `config.json` 中的路径正确，相关文件已放置到指定位置。

- **图片未被处理**  
  请确认图片格式受支持，且已放入 `input_dir` 指定的文件夹。

- **输出目录无结果**  
  检查程序运行日志，确认无报错。如有错误图片，会自动移动到 `output_captions/errors` 文件夹。

- **依赖缺失或程序无法启动**  
  Nuitka 打包已集成所有依赖，无需单独安装 Python。若仍无法运行，请联系开发者。

---

## 配置文件示例（config.json）

```json
{
  "input_dir": "./input_images",
  "output_dir": "./output_captions",
  "model_path": "./path_to_model",
  "tokenizer_path": "./path_to_tokenizer",
  "model_params": {
    "latent_dim": 512,
    "nhead": 8,
    "num_layers": 6,
    "dim_feedforward": 2048,
    "max_seq_length": 128
  },
  "image_preprocessing": {
    "resize": [ 224, 224 ],
    "mean": [ 0.485, 0.456, 0.406 ],
    "std": [ 0.229, 0.224, 0.225 ]
  },
  "generation": {
    "max_length": 30,
    "beam_size": 5,
    "temperature": 1.0
  },
  "device": "cuda"
}
```

---

## 目录结构示例

C:\Program Files\ImageCaptioning\
│
├── ImageCaptioning.exe
├── config.json
├── best_model_xxx.pth
├── bert-base-uncased\  （或自定义分词器文件夹）
├── input_images\
│   ├── img1.jpg
│   └── img2.png
├── output_captions\
│   ├── img1.txt
│   └── img2.txt
└── ...

---

如需插入命令行示例，请使用如下格式：

```bash
ImageCaptioning.exe --config config.json
```

---

## 许可证

本软件仅供学术研究与学习使用，禁止用于商业用途。模型权重及数据集版权归原作者所有。

---

如有问题，请联系开发者或提交反馈。
