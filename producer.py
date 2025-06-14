import os
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import PIL.Image as Image
from transformers import BertTokenizer, BertForMaskedLM
import torch.nn as nn

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载预训练的ResNet50模型
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层
resnet.to(device)  # 将模型移动到GPU

# 加载BERT模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
bert_model.to(device)  # 将模型移动到GPU

# 定义一个类来将图像特征与BERT的输入结合
class ImageTextModel(torch.nn.Module):
    def __init__(self, image_encoder, bert_model):
        super(ImageTextModel, self).__init__()
        self.image_encoder = image_encoder
        self.bert = bert_model
        self.image_feature_dim = 2048  # ResNet50的输出特征维度
        self.text_feature_dim = 768  # BERT的输出特征维度
        self.combined_feature_dim = 1536  # 拼接后的特征维度

        # 定义一个线性层将图像特征映射到与文本特征相同的维度
        self.image_projection = nn.Linear(self.image_feature_dim, self.text_feature_dim).to(device)

        # 获取BERT的嵌入层
        self.bert_embeddings = self.bert.bert.embeddings

        # 定义一个线性层将拼接后的特征维度从1536降回到768
        self.combined_projection = nn.Linear(self.combined_feature_dim, self.text_feature_dim).to(device)

    def forward(self, image_input, text_input):
        # 获取图像特征
        img_features = self.image_encoder(image_input)  # (batch_size, 2048, 1, 1)
        img_features = img_features.view(img_features.size(0), -1)  # (batch_size, 2048)
        img_features = self.image_projection(img_features)  # (batch_size, 768)

        # 将图像特征扩展到文本序列长度
        img_features = img_features.unsqueeze(1)  # (batch_size, 1, 768)
        img_features = img_features.repeat(1, text_input.size(1), 1)  # (batch_size, sequence_length, 768)

        # 将文本输入转换为嵌入向量
        text_embeddings = self.bert_embeddings(text_input)  # (batch_size, sequence_length, 768)

        # 将图像特征与文本嵌入拼接
        combined_input = torch.cat((img_features, text_embeddings), dim=-1)  # (batch_size, sequence_length, 1536)

        # 使用线性层将拼接后的特征维度从1536降回到768
        combined_input = self.combined_projection(combined_input)  # (batch_size, sequence_length, 768)

        # 输入BERT模型
        outputs = self.bert(inputs_embeds=combined_input)

        return outputs

# 加载模型
def load_model(model_path, device):
    model = ImageTextModel(resnet, bert_model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式
    return model

# 对图像进行标注
def annotate_image(model, image_path, max_length):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # 变为(batch_size, 3, 224, 224)并移动到GPU
    
    # 初始化生成的句子
    generated_sentence = []
    current_input = tokenizer("", return_tensors="pt").to(device)

    for _ in range(max_length):
        # 将已生成的部分作为上下文输入
        inputs = tokenizer(" ".join(generated_sentence), return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        inputs['input_ids'] = torch.cat((inputs['input_ids'], current_input['input_ids']), dim=1)
        inputs['attention_mask'] = torch.cat((inputs['attention_mask'], current_input['attention_mask']), dim=1)

        # 进行推理
        with torch.no_grad():
            outputs = model(img_tensor, inputs['input_ids'])

        # 获取下一个词的概率分布
        logits = outputs.logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probabilities, dim=-1)

        # 将生成的词添加到句子中
        generated_sentence.append(tokenizer.decode(next_token, skip_special_tokens=True))

        # 更新输入
        current_input = tokenizer(" ".join(generated_sentence), return_tensors="pt").to(device)

    return " ".join(generated_sentence)

# 主函数
def main(config):
    # 读取配置文件
    with open(config, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    model_folder = config_data['model_folder']
    image_folder = config_data['image_folder']
    output_file = config_data['output_file']
    max_length = config_data['max_length']

    # 找到最新的模型文件
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.pth')]
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_folder, x)), reverse=True)
    latest_model_path = os.path.join(model_folder, model_files[0])

    # 加载模型
    model = load_model(latest_model_path, device)

    # 对图像文件夹中的所有图片进行标注
    annotations = {}
    for image_file in os.listdir(image_folder):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            annotation = annotate_image(model, image_path, max_length)
            annotations[image_file] = annotation

    # 将标注结果保存到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)

    print(f"Annotations saved to {output_file}")

if __name__ == "__main__":
    config_file = "config.json"
    main(config_file)