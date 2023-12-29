import random
import sys
import torch
from fine_tune import fine_tune_clip_model, Flickr8kDataset
from transformers import BertTokenizer
from eval import evaluate_performance
import logging
import clip
from torchvision import transforms

logging.basicConfig(filename='logfile.log', level=logging.INFO)
best_score = float('-inf')
best_params = {}
best_model = None

# 设定 Bert 模型路径和 tokenizer
bert_model_path = r"E:\pythonProject\bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
# 数据集和文件路径
data_dir = 'datasets'
captions_file = 'datasets\en_train.json'
# 对图像进行处理和转换的方法
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.RandomRotation(15),  # 随机旋转
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 色彩变换
    transforms.CenterCrop(224),  # 中心裁剪
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像标准化
])
# 创建数据集对象
dataset = Flickr8kDataset(data_dir, captions_file, transform, bert_tokenizer)
print("Data preparation completed. Starting fine-tuning.")
sys.stdout.flush()

# 加载预训练的 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
# 定义超参数
learning_rates = [0.0001, 0.001, 0.0005]
batch_sizes = [16, 32, 64, 128]
epochs_list = [4, 5, 6, 8]
# 定义损失函数和混合精度
loss_fn = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# 微调的迭代次数
num_iterations = 10
for i in range(num_iterations):
    # 随机选择超参数
    learning_rate = random.choice(learning_rates)
    batch_size = random.choice(batch_sizes)
    epochs = random.choice(epochs_list)

    print(f"Hyperparameters: learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")
    logging.info(f"Current learning_rate: {learning_rate}, batch_size: {batch_size}, epochs: {epochs}")

    # 微调 CLIP 模型
    fine_tuned_model = fine_tune_clip_model(clip_model, dataset, loss_fn, scaler, device, learning_rate, batch_size,
                                            epochs)

    # 为每个模型创建一个唯一的文件名，以迭代次数作为后缀
    model_name = f"fine_tuned_model_{i + 1}.pth"

    # 保存训练好的模型
    torch.save(fine_tuned_model.state_dict(), model_name)
    fine_tuned_model.load_state_dict(torch.load(model_name))
    fine_tuned_model.eval()  # 设置模型为评估模式

    # 评估微调后的模型
    val_metric = evaluate_performance(fine_tuned_model)
    print(f"Hyperparameter: learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")
    print(f"Validation Metric: {val_metric}")
    # 将评估指标记录到日志文件中
    logging.info(
        f"Validation Metric for learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}: {val_metric}")
    # 如果验证分数优于最佳分数，则更新最佳分数和模型参数
    if val_metric["r_mean"] > best_score:
        best_score = val_metric["r_mean"]
        best_params = {'learning_rate': learning_rate, 'batch_size': batch_size, 'epochs': epochs}
        best_model = fine_tuned_model
        torch.save(best_model.state_dict(), 'best_fine_tuned_clip_model.pth')

print('Best hyperparameters:')
print(best_params)
print('Best validation score:', best_score)

best_fine_tuned_model = best_model
