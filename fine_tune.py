# 导入所需库
from typing import Any
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import torch.nn.functional as F
import os
from transformers import BertTokenizer
import torch
from tqdm import tqdm


class Flickr8kDataset(Dataset):
    """
        flickr8k 数据集类

        Args:
        - image_dir (str): 图像所在目录
        - captions_file (str): 描述文件路径
        - transform (callable, optional): 图像转换方法
        - bert_tokenizer (BertTokenizer, optional): BertTokenizer 对象
    """
    def __init__(self, image_dir, captions_file, transform=None, bert_tokenizer=None):
        self.image_dir = image_dir
        self.image_dir = image_dir
        self.transform = transform
        self.bert_tokenizer = bert_tokenizer

        with open(captions_file, 'r') as f:
            self.data = json.load(f)

        # Extract image paths and captions
        self.image_paths = [item['image'] for item in self.data]
        self.captions = [item['caption'] for item in self.data]

    def __len__(self):
        """
            返回数据集长度（图像数量）
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
              根据索引返回对应图像和描述

              Args:
              - index (int): 图像索引

              Returns:
              - image (PIL.Image.Image): 图像
              - captions_list (list): 描述列表
              """
        image_path = self.image_paths[index]
        full_image_path = os.path.join(self.image_dir, image_path)
        image = Image.open(full_image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        captions = self.captions[index]  # Get captions directly using index
        captions_list = [str(caption) for caption in captions if isinstance(caption, str)]

        """print(f"Sample {index + 1}: Image Size - {image.size()}, Captions - {captions}")"""
        return image, captions_list


# 定义保存和加载模型的函数
def save_checkpoint(clip_model, optimizer, epoch, loss, filename):
    """
      保存模型和优化器检查点

      Args:
      - model (nn.Module): 模型
      - optimizer (torch.optim.Optimizer): 优化器
      - epoch (int): 当前周期
      - loss (float): 损失
      - filename (str): 检查点文件名
      """
    torch.save({
        'epoch': epoch + 1,
        'clip_model_state_dict': clip_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, filename)


def load_checkpoint(clip_model, optimizer, filename):
    """
       加载模型和优化器检查点

       Args:
       - model (nn.Module): 模型
       - optimizer (torch.optim.Optimizer): 优化器
       - filename (str): 检查点文件名

       Returns:
       - model (nn.Module): 加载后的模型
       - optimizer (torch.optim.Optimizer): 加载后的优化器
       - epoch (int): 上次周期数
       - loss (float): 上次损失
       """
    checkpoint = torch.load(filename)
    clip_model.load_state_dict(checkpoint['clip_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return clip_model, optimizer, epoch, loss


def fine_tune_clip_model(clip_model, dataset, loss_fn, scaler, device, lr, batch_size, epochs):
    """
        微调 CLIP 模型

        Args:
        - model (nn.Module): CLIP 模型
        - dataset (Flickr8kDataset): 自定义数据集
        - loss_fn (callable): 损失函数
        - scaler (torch.cuda.amp.GradScaler, optional): 混合精度训练器
        - device (str): 训练设备 ('cuda' 或 'cpu')
        - lr (float): 学习率
        - batch_size (int): 批量大小
        - epochs (int): 训练周期数

        Returns:
        - model (nn.Module): 微调后的模型
        """
    clip_model.train()  # 设置模型为训练模式
    optimizer = torch.optim.Adam(clip_model.parameters(), lr)
    bert_model_path = r"E:\pythonProject\bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    data_loader: DataLoader[Any] = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_batches = len(data_loader)
    total_samples = len(data_loader.dataset)
    # 循环遍历训练集多个 epochs
    start_epoch = 0  # 设置从第几个周期继续训练
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0  # 记录每一 epoch 总损失
        # 遍历数据加载器中的批次数据
        for batch_idx, (batch_images, batch_captions) in enumerate(data_loader):
            optimizer.zero_grad()

            # 对文本进行填充或截断处理
            padded_captions = []
            for captions in batch_captions:
                # 对于每个caption，使用BERT tokenizer进行编码处理
                tokens = [bert_tokenizer.encode(caption, add_special_tokens=True, max_length=77, padding='max_length',
                                                truncation=True) for caption in captions]
                # 将编码后的tokens添加到padded_captions列表中
                padded_captions.append(tokens)
            max_len = max(len(tokens) for tokens in padded_captions)
            # 遍历每个token序列，并根据最大长度进行填充或截断处理
            padded_captions = [tokens + [0] * (max_len - len(tokens)) for tokens in padded_captions]

            # 将数据转换为 PyTorch 张量并传入模型
            text_inputs = torch.tensor(padded_captions).to(device)
            image_inputs = batch_images.to(device)

            # 编码图像和文本
            with torch.cuda.amp.autocast():
                image_features = clip_model.encode_image(image_inputs)
                text_features_list = []
                for text_input in text_inputs:
                    # 提取文本特征
                    with torch.no_grad():
                        text_features = clip_model.encode_text(text_input).float()
                    text_features_list.append(text_features)
                # 求平均
                mean_text_features = torch.mean(torch.stack(text_features_list), dim=0)

                # 计算 logits 和 loss
                logits_per_image = 100.0 * F.linear(mean_text_features, image_features)
                targets = torch.arange(len(batch_images)).to(device)
                loss = loss_fn(logits_per_image, targets)

            epoch_loss += loss.item()

            # 更新训练进度
            progress = ((epoch * total_batches) + batch_idx) / (epochs * total_batches)
            tqdm.write(f"Epoch {epoch + 1}, Total Progress: {progress * 100:.2f}%")

            # 反向传播和更新模型参数
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            # 保存模型检查点
            save_checkpoint(clip_model, optimizer, epoch, loss, f"checkpoints/clip_model_epoch_{epoch + 1}.pth")

        # 计算并打印每个 epoch 的平均损失
        average_epoch_loss = epoch_loss / total_samples
        print(f"Epoch {epoch + 1} Average Loss: {average_epoch_loss:.4f}")
    # 保存微调后的模型
    torch.save(clip_model.state_dict(), 'fine_tuned_clip_model.pth')
    return clip_model  # 在函数结束前返回微调后的模型
