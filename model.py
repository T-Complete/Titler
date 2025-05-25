import torch.nn as nn
import torch
import torchvision.models as models
from transformers import BertTokenizer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class CNNModel(nn.Module):
    def __init__(self, embedding_size):
            super().__init__()
            resnet = models.resnet152(pretrained=True)
            modules = list(resnet.children())[:-1]  # 移除最后的全连接层
            self.resnet_module = nn.Sequential(*modules)
            self.linear = nn.Linear(resnet.fc.in_features, embedding_size)
            
    def forward(self, x):
            with torch.no_grad():  # 冻结ResNet参数
                features = self.resnet_module(x)
            features = features.flatten(1)
            return self.linear(features)

class TextDecoder(nn.Module):
    """
    基于Transformer的文本解码器
    功能：
    - 将编码向量转换为文本序列
    - 支持自注意力机制和编码器-解码器注意力
    - 可调节的深度和注意力头数
    """
    def __init__(self,
                 latent_dim: int = 512,
                 vocab_size: int = 30522,  # 与BERT tokenizer保持一致
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 max_seq_length: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        # 词嵌入层（与BERT的嵌入维度保持一致）
        self.embedding = nn.Embedding(vocab_size, latent_dim)
        
        # 位置编码
        self.positional_encoding = nn.Parameter(
            torch.randn(max_seq_length, 1, latent_dim)
        )
        
        # Transformer解码层
        decoder_layer = TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(latent_dim, vocab_size)
        
        # 初始化参数
        self._init_weights()
        self.max_seq_length = max_seq_length

    def _init_weights(self):
        """初始化权重（与BERT风格一致）"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, 
                encoded_vec: torch.Tensor,  # 编码向量 [batch_size, latent_dim]
                tgt_seq: torch.Tensor,      # 目标序列 [batch_size, seq_len]
                tgt_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None):
        """
        前向传播流程：
        1. 将编码向量扩展为序列形式
        2. 处理目标序列嵌入和位置编码
        3. 通过Transformer解码器
        4. 生成词汇概率分布
        """
        batch_size = encoded_vec.size(0)
        
        # 将编码向量扩展为 (seq_len, batch_size, latent_dim)
        memory = encoded_vec.unsqueeze(1)  # [batch_size, 1, latent_dim]
        memory = memory.expand(-1, self.max_seq_length, -1)  # [batch_size, max_seq_length, latent_dim]
        memory = memory.permute(1, 0, 2)  # [max_seq_length, batch_size, latent_dim]
        
        # 处理目标序列嵌入
        tgt = self.embedding(tgt_seq) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32))
        tgt = tgt.permute(1, 0, 2)  # [seq_len, batch_size, latent_dim]
        tgt = tgt + self.positional_encoding[:tgt.size(0), :]
        
        # 生成注意力mask（防止看到未来信息）
        if tgt_mask is None:
            seq_len = tgt.size(0)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            tgt_mask = tgt_mask.to(encoded_vec.device)
        
        # 通过Transformer解码器
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # 输出层处理
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, latent_dim]
        logits = self.output_layer(output)
        
        return logits  # [batch_size, seq_len, vocab_size]

    def generate(self,
                 encoded_vec: torch.Tensor,
                 start_token_id: int = 101,  # [CLS] token
                 end_token_id: int = 102,   # [SEP] token
                 max_length: int = 128,
                 temperature: float = 1.0,
                 beam_size: int = 5):
        """
        文本生成方法（使用束搜索）
        参数：
        encoded_vec: 编码向量 [1, latent_dim]
        beam_size: 束搜索宽度
        """
        # 确保在评估模式
        self.eval()
        
        # 初始化束搜索
        sequences = [[start_token_id], ]
        scores = [0.0]
        
        for _ in range(max_length):
            all_candidates = []
            all_finished = True
            for i in range(len(sequences)):
                seq = sequences[i]
                current_score = scores[i]
                
                # 跳过已生成结束符的序列
                if seq[-1] == end_token_id or len(seq) >= max_length:
                    all_candidates.append((seq, current_score))
                    continue
                
                #还有序列需要扩展
                all_finished = False
                # 准备输入
                input_tensor = torch.tensor([seq], device=encoded_vec.device)
                # 前向传播
                with torch.no_grad():
                    logits = self.forward(
                        encoded_vec,
                        input_tensor
                    )  # [1, seq_len, vocab_size]
                
                # 获取最后一个token的logits
                next_logits = logits[0, -1, :] / temperature
                next_probs = torch.softmax(next_logits, dim=-1)
                topk_probs, topk_ids = torch.topk(next_probs, beam_size)
                
                # 扩展候选序列
                for j in range(beam_size):
                    candidate_seq = seq + [topk_ids[j].item()]
                    candidate_score = current_score + torch.log(topk_probs[j]).item()
                    all_candidates.append((candidate_seq, candidate_score))
            
            # 选择top-k候选
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences, scores = zip(*ordered[:beam_size])
            sequences = list(sequences)
            scores = list(scores)

            if all_finished:
                break
        
        # 选择最佳序列
        best_seq = sequences[0]
        # 去除后续的结束符
        if end_token_id in best_seq:
            end_pos = best_seq.index(end_token_id)
            best_seq = best_seq[:end_pos+1]
        
        return best_seq

class ImageCaptioningModel(nn.Module):
    """完整的图像描述生成模型"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder  # 图像编码器（如ResNet）
        self.decoder = decoder  # 文本解码器
    
    def forward(self, images, captions):
        # 图像编码
        encoded_vecs = self.encoder(images)  # [batch_size, latent_dim]
        
        # 文本解码
        logits = self.decoder(encoded_vecs, captions)
        return logits

def test_model_integration():
    """模型完整性测试方法，验证以下内容：
    1. 前向传播的数据流是否正常
    2. 生成过程是否能产生有效输出
    3. 各组件形状匹配性
    """
    # 配置参数
    batch_size = 2
    vocab_size = 30522
    max_seq_len = 20
    latent_dim = 512
    # 初始化模型组件
    encoder = CNNModel(latent_dim)
    decoder = TextDecoder(latent_dim=latent_dim)
    model = ImageCaptioningModel(encoder, decoder)
    # 测试前向传播 ------------------------------------------------
    # 生成随机测试数据
    dummy_images = torch.randn(batch_size, 3, 224, 224)  # 符合ResNet输入尺寸
    dummy_captions = torch.randint(0, vocab_size, (batch_size, max_seq_len))

    # 运行前向传播
    logits = model(dummy_images, dummy_captions)
    
    # 验证输出形状
    assert logits.shape == (batch_size, max_seq_len, vocab_size), \
        f"前向传播输出形状错误，期望{(batch_size, max_seq_len, vocab_size)}，实际{logits.shape}"
    # 测试生成过程 ------------------------------------------------
    # 生成随机编码向量（模拟编码器输出）
    dummy_encoded = torch.randn(1, latent_dim)  # 单样本测试
    
    # 运行生成方法
    generated_ids = decoder.generate(dummy_encoded, max_length=max_seq_len)
    
    # 验证生成结果有效性
    assert isinstance(generated_ids, list), "生成结果应为列表"
    assert len(generated_ids) > 1, "生成序列过短"

    # 使用tokenizer解码
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print("="*50)
    print("测试结果：")
    print(f"前向传播输出形状验证通过: {logits.shape}")
    print(f"生成token数量: {len(generated_ids)}")
    print(f"生成文本示例: {decoded_text[:50]}...")  # 显示前50个字符
    print("="*50)