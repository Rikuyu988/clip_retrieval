import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataloader import ClipDataset
from utils.metrics import itm_eval  # 自定义的评估函数，根据具体情况导入
import clip


# 定义评估函数
def evaluate_performance(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.to(device)
    model.eval()

    i_features = []
    t_features = []

    for iteration, batch in tqdm(enumerate(test_loader)):
        images, texts = batch
        images = images.to(device)

        # 处理文本数据
        tokenized_texts = clip.tokenize(texts).to(device)

        with torch.no_grad():
            images_feature = model.encode_image(images).float()
            i_features.append(images_feature)
            texts_feature = model.encode_text(tokenized_texts).float()
            t_features.append(texts_feature)

    i_features = torch.cat(i_features, 0)
    t_features = torch.cat(t_features, 0)

    # 转换为张量后进行归一化操作
    i_features = i_features / i_features.norm(dim=-1, keepdim=True)
    t_features = t_features / t_features.norm(dim=-1, keepdim=True)

    # Calculate similarity scores
    logins_per_image = i_features @ t_features.t()
    logins_per_text = logins_per_image.t()

    logins_per_image = logins_per_image.cpu().numpy()
    logins_per_text = logins_per_text.cpu().numpy()

    # Evaluate the features
    results = itm_eval(logins_per_image, logins_per_text, test_loader.dataset.txt2img,
                       test_loader.dataset.img2txt)

    return results


# 在这里加载测试数据集
datasets_path = "datasets/"
datasets_val_json_path = "datasets/en_val.json"
test_val_lines = json.load(open(datasets_val_json_path, mode='r', encoding='utf-8'))
test_dataset = ClipDataset([224, 224], test_val_lines, datasets_path, random=False)
batch_size = 64
num_workers = 4

# 在这里加载预训练的 CLIP 模型
clip_model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
clip_model.load_state_dict(torch.load('fine_tuned_model_1.pth'))

# 进行评估
evaluation_results = evaluate_performance(clip_model)
print(evaluation_results)
