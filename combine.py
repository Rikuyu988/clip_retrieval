from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
import torch.nn.functional as F
from combiner import Combiner
from fine_tune import Flickr8kDataset
import torch
import clip
from eval import evaluate_performance

# 在数据集类中对图像进行处理和转换
transform = transforms.Compose([
    transforms.Resize(256),  # 调整图像大小
    transforms.CenterCrop(224),  # 中心裁剪
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])
bert_model_path = r"E:\pythonProject\bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)

# 训练函数
def train_combiner(clip_model, combiner_model, train_loader, criterion, optimizer, device, num_epochs=10):
    combiner_model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, captions_list, labels in train_loader:
            images = images.to(device)
            # text_inputs = clip.tokenize(list(captions_list)).to(device)
            padded_captions = []
            for captions in captions_list:
                tokens = [bert_tokenizer.encode(caption, add_special_tokens=True, max_length=77, padding='max_length',
                                                truncation=True) for caption in captions]
                padded_captions.append(tokens)
            max_len = max(len(tokens) for tokens in padded_captions)
            padded_captions = [tokens + [0] * (max_len - len(tokens)) for tokens in padded_captions]

            # 将数据转换为 PyTorch 张量并传入模型
            text_inputs = torch.tensor(padded_captions).to(device)
            # text_inputs = clip.tokenize([str(caption) for caption in captions_list]).to(device)

            # 提取图像特征
            with torch.no_grad():
                image_features = clip_model.encode_image(images).float()

            # 提取文本特征
            for text_input in text_inputs:
                # 提取文本特征
                with torch.no_grad():
                    text_features = clip_model.encode_text(text_input).float()

            # 前向传播
            optimizer.zero_grad()
            outputs = combiner_model(image_features, text_features)
            loss = criterion(outputs, labels)
            # loss = torch.nn.CrossEntropyLoss()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}")



# 通过 clip.load() 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")
clip_model.eval()  # 设置模型为评估模式
# 在 fine-tuning 后保存微调后的模型
torch.save(clip_model.state_dict(), 'fine_tuned_model_1.pth')

learning_rate = 0.001
batch_size = 32
num_epochs = 0
model_save_path = "model_combiner.pth"  # 设置保存路径

# 加载权重文件到 model 实例
clip_model.load_state_dict(torch.load('fine_tuned_model_1.pth'))
print('CLIP model loaded successfully')

clip_feature_dim = 512
projection_dim = 128
hidden_dim = 128
# 创建Combiner模型实例
combiner = Combiner(clip_feature_dim,projection_dim,hidden_dim).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(combiner.parameters(), lr=learning_rate)

# 创建数据集和数据加载器（假设使用了自定义的数据集类）
data_dir = 'datasets'
captions_file = 'datasets\en_train.json'
train_dataset = Flickr8kDataset(data_dir, captions_file, transform, bert_tokenizer)  # 使用你的数据集类初始化训练数据集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练Combiner模型
train_combiner(clip_model, combiner, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

# 保存训练好的模型
torch.save(combiner.state_dict(), model_save_path)
"""
val_metric = evaluate_performance(combiner)
print(f"Validation Metric: {val_metric}")
"""

