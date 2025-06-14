import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torch import nn
from transformers import AutoModelForMaskedLM
from torchvision.models import ResNet50_Weights
import datetime
import queue
import threading
import time
import os

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 载入COCO数据集
class CocoDataset(Dataset):
    def __init__(self, image_dir, annotation_file, tokenizer, max_length=128, split='train'):
        self.coco = COCO(annotation_file)  # 加载标注文件
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # 获取所有图像ID
        self.image_ids = list(self.coco.imgs.keys())
        
        # 如果是测试集，只取前100个图像ID
        if split == 'test':
            self.image_ids = self.image_ids[:100]
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # 获取当前图像ID
        image_id = self.image_ids[idx]
        
        # 获取图像路径和描述
        image_info = self.coco.imgs[image_id]
        image_path = f"{self.image_dir}/{image_info['file_name']}"
        
        # 读取图像
        image = Image.open(image_path).convert("RGB")
        image = transform(image)  # 应用图像预处理
        
        # 获取图像的描述（可以取多个描述，这里选取第一个）
        caption_ids = self.coco.getAnnIds(imgIds=image_id)
        captions = self.coco.loadAnns(caption_ids)
        caption = captions[0]['caption']
        
        # 使用tokenizer将文本描述转换为token
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

def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    captions = [item['caption'] for item in batch]
    
    return {
        'images': images,
        'input_ids': input_ids,
        'attention_masks': attention_masks,
        'image_paths': image_paths,
        'captions': captions
    }

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
        self.image_projection = nn.Linear(self.image_feature_dim, self.text_feature_dim)

        # 获取BERT的嵌入层
        self.bert_embeddings = self.bert.bert.embeddings

        # 定义一个线性层将拼接后的特征维度从1536降回到768
        self.combined_projection = nn.Linear(self.combined_feature_dim, self.text_feature_dim)

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

    def compute_loss(self, image_input, text_input, target_ids, criterion):
        outputs = self.forward(image_input, text_input)
        logits = outputs.logits[:, :-1, :].contiguous()
        target_ids = target_ids[:, 1:].contiguous()
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        return loss

def extract_image_features(image, model):
    img_features = model.image_encoder(image)  # (batch_size, 2048, 1, 1)
    img_features = img_features.view(img_features.size(0), -1)  # (batch_size, 2048)
    img_features = model.image_projection(img_features)  # (batch_size, 768)
    return img_features

def generate_caption_beam_search(image, model, tokenizer, device, beam_size=3, max_length=50, top_k=50, top_p=0.95, repetition_penalty=1.2):
    model.eval()
    with torch.no_grad():
        # 获取图像特征
        img_features = extract_image_features(image, model)
        img_features = img_features.unsqueeze(1)  # (batch_size, 1, 768)
        
        # 初始化束搜索
        beam = [{'sequence': torch.tensor([tokenizer.cls_token_id], dtype=torch.long).to(device), 'score': 0.0, 'visited': set([tokenizer.cls_token_id])}]
        
        for _ in range(max_length):
            new_beam = []
            for beam_item in beam:
                sequence = beam_item['sequence']
                score = beam_item['score']
                visited = beam_item['visited']
                
                if sequence[-1] == tokenizer.sep_token_id:
                    new_beam.append(beam_item)
                    continue
                
                inputs = {'inputs_embeds': img_features, 'attention_mask': torch.ones(1, sequence.size(0)).to(device)}
                outputs = model.bert(**inputs)
                logits = outputs.logits[:, -1, :]
                
                # 应用 Top-K 或 Top-P 抽样
                if top_k > 0:
                    topk_log_probs, topk_ids = torch.topk(logits, top_k)
                elif top_p > 0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = -float('Inf')
                    topk_log_probs, topk_ids = torch.topk(torch.log_softmax(logits, dim=-1), beam_size)
                
                # 施加重复惩罚
                for i in range(beam_size):
                    if topk_ids[0][i].item() in visited:
                        logits[0][topk_ids[0][i]] /= repetition_penalty  # 降低重复词汇的概率
                topk_log_probs, topk_ids = torch.topk(logits, beam_size)
                
                for i in range(beam_size):
                    new_sequence = torch.cat((sequence, topk_ids[0][i].unsqueeze(0)))
                    new_score = score + topk_log_probs[0][i].item()
                    new_visited = visited.copy()
                    new_visited.add(topk_ids[0][i].item())
                    new_beam.append({'sequence': new_sequence, 'score': new_score, 'visited': new_visited})
            
            beam = sorted(new_beam, key=lambda x: x['score'], reverse=True)[:beam_size]
        
        best_sequence = beam[0]['sequence']
        predicted_text = tokenizer.decode(best_sequence, skip_special_tokens=True)
        
        return predicted_text

def visualize_test(test_dataset, model, device, tokenizer):
    plt.clf()
    plt.cla()
    plt.ion()
    
    # 从测试数据集中随机选择一个样本
    idx = torch.randint(0, len(test_dataset), (1,)).item()
    sample = test_dataset[idx]
    
    # 检查图像文件是否存在
    image_path = sample['image_path']
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return
    
    # 将图像输入移动到设备
    image = sample['image'].unsqueeze(0).to(device)
    
    # 使用 Beam Search 生成描述
    predicted_text = generate_caption_beam_search(image, model, tokenizer, device, beam_size=3, max_length=50)
    
    # 可视化输入图像与预测结果
    img = Image.open(image_path).convert("RGB")
    plt.imshow(img)
    plt.title(f"Original: {sample['caption']}\nPredicted: {predicted_text}")
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    plt.pause(10)  # 暂停10秒，确保图像有足够的时间显示
    plt.ioff()
    plt.close()

q=queue.Queue()
def timing(q):
    time.sleep(3600)
    q.put("finished")
t=threading.Thread(target=timing,args=(q,))
def train_model(num_epochs=10, dataset=None, save_model=True, num_worker=0):
    dataiter = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_worker, collate_fn=collate_fn,
                          persistent_workers=True if num_worker>1 else False,pin_memory=True if torch.cuda.is_available() else False)
    t.start()
    cnt = 0
    msg="none"
    while True:
        cnt += 1
        for epoch in range(num_epochs):
            image_text_model.train()
            total_loss = 0
            for batch in dataiter:
                images = batch['images'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_masks = batch['attention_masks'].to(device)

                # 获取图像特征
                img_features = extract_image_features(images, image_text_model)

                # 使用束搜索生成序列
                generated_sequences = []
                for i in range(images.size(0)):
                    generated_sequence = generate_caption_beam_search(images[i].unsqueeze(0), image_text_model, tokenizer, device)
                    generated_sequences.append(generated_sequence)

                # 将生成的序列转换为token
                generated_inputs = tokenizer(generated_sequences, return_tensors="pt", padding=True, truncation=True, max_length=input_ids.size(1))
                generated_input_ids = generated_inputs['input_ids'].to(device)
                generated_attention_masks = generated_inputs['attention_mask'].to(device)

                # 确保生成的序列和目标序列的长度一致
                if generated_input_ids.size(1) < input_ids.size(1):
                    padding = torch.full((generated_input_ids.size(0), input_ids.size(1) - generated_input_ids.size(1)), tokenizer.pad_token_id).to(device)
                    generated_input_ids = torch.cat((generated_input_ids, padding), dim=1)
                elif generated_input_ids.size(1) > input_ids.size(1):
                    generated_input_ids = generated_input_ids[:, :input_ids.size(1)]

                # 计算损失
                loss = image_text_model.compute_loss(images, generated_input_ids, input_ids, criterion)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                msg=q.get()
                if(msg=="finished"):
                    break

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataiter)}")
            if(msg=="finished"):
                break

            # 可视化测试 - 使用 COCO 测试数据集中的一个样本
            visualize_test(test_dataset, image_text_model, device, tokenizer)

        # 用户选择继续训练模型还是保存当前模型
        choice = input("Continue training (Y/n)? ")
        if choice.lower() == 'n':
            if save_model:
                try:
                    # 获取当前时间
                    now = datetime.datetime.now()
                    date_str = now.strftime("%Y%m%d")
                    time_str = now.strftime("%H%M%S")

                    # 构建文件名
                    #filename = f"BertRes101_{dataset}_{date_str}_{time_str}_{cnt * num_epochs}.pth"
                    filename = f"BertRes101_{date_str}_{time_str}_{cnt * num_epochs}.pth"
                    model_path = "E:\\ANNprogram\\model"
                    model_path=os.path.join(model_path,filename)

                    # 确保目录存在
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)

                    # 尝试保存模型
                    torch.save(image_text_model.state_dict(), model_path)
                    print(f"模型成功保存到 {model_path}")

                except Exception as e:
                    # 捕获异常并打印错误信息
                    print(f"保存模型到指定路径失败: {e}")
                    try:
                        # 尝试保存到工作目录
                        working_dir_model_path = os.path.join(os.getcwd(), filename)
                        torch.save(image_text_model.state_dict(), working_dir_model_path)
                        print(f"模型已成功保存到工作目录: {working_dir_model_path}")
                    except Exception as backup_error:
                        print(f"保存模型到工作目录也失败: {backup_error}")
                visualize_test(test_dataset=test_dataset, model=image_text_model, tokenizer=tokenizer, device=device)
            #os.system("shutdown -s -t 15")
            break

if __name__ == '__main__':
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载预训练的ResNet50模型
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层
    resnet.to(device)  # 将模型移动到GPU

    # 加载预训练的BERT模型
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    bert_model.to(device)  # 将模型移动到GPU
    # 用户选择加载已保存模型还是新模型
    choice = input("Load existing model (Y/n)? ")
    if choice.lower() == 'y':
        model_path = input("Enter the path to the model: ")
        image_text_model = ImageTextModel(resnet, bert_model).to(device)
        image_text_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        image_text_model = ImageTextModel(resnet, bert_model).to(device)
    # 创建COCO测试数据集实例
    test_image_dir = 'E:\\ANNprogram\\data\\coco\\Images\\val2017'
    test_annotation_file = 'E:\\ANNprogram\\data\\coco\\Annotations\\captions_val2017.json'
    test_dataset = CocoDataset(test_image_dir, test_annotation_file, tokenizer, split='test')
    image_dir = 'E:\\ANNprogram\\data\\coco\\Images\\train2017'
    annotation_file = 'E:\\ANNprogram\\data\\coco\\Annotations\\captions_train2017.json'
    dataset = CocoDataset(image_dir, annotation_file, tokenizer)
    # 定义损失函数和优化器
    optimizer = torch.optim.AdamW(image_text_model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # visualize_test(test_dataset=test_dataset, model=image_text_model, tokenizer=tokenizer, device=device)
    # 调用训练
    train_model(num_epochs=10, dataset=dataset, num_worker=16)