import matplotlib.pyplot as plt

# 模拟不同模型的评估指标
model_results = [
    {
        'txt_r1': 25.3,
        'txt_r5': 31.4,
        'txt_r10': 77.9,
        'txt_r_mean': 46.67,
        'img_r1': 39.4,
        'img_r5': 65.8,
        'img_r10': 74.1,
        'img_r_mean': 59.33,
        'r_mean': 53.0
    },
    {
        'txt_r1': 31.2,
        'txt_r5': 49.5,
        'txt_r10': 81.0,
        'txt_r_mean': 50.6,
        'img_r1': 49.4,
        'img_r5': 71.3,
        'img_r10': 78.3,
        'img_r_mean': 73.33,
        'r_mean': 61.67
    },
    {
        'txt_r1': 46.4,
        'txt_r5': 71.6,
        'txt_r10': 82.1,
        'txt_r_mean': 66.7,
        'img_r1': 35.44,
        'img_r5': 66.3,
        'img_r10': 77.36,
        'img_r_mean': 59.699999999999996,
        'r_mean': 63.2
    },
    {'txt_r1': 48.4,
     'txt_r5': 72.6,
     'txt_r10': 81.4,
     'txt_r_mean': 67.46666666666667,
     'img_r1': 37.56,
     'img_r5': 66.84,
     'img_r10': 77.74,
     'img_r_mean': 60.71333333333333,
     'r_mean': 64.09
     },
    {'txt_r1': 48.6,
     'txt_r5': 72.2,
     'txt_r10': 82.0,
     'txt_r_mean': 67.60000000000001,
     'img_r1': 37.7,
     'img_r5': 68.48,
     'img_r10': 79.64,
     'img_r_mean': 61.94,
     'r_mean': 64.77000000000001
     },
    {'txt_r1': 77.8,
     'txt_r5': 93.0,
     'txt_r10': 96.8,
     'txt_r_mean': 89.2,
     'img_r1': 63.14,
     'img_r5': 88.1,
     'img_r10': 94.24,
     'img_r_mean': 81.82666666666667,
     'r_mean': 85.51333333333334
     },
    {'txt_r1': 72.3,
     'txt_r5': 90.3,
     'txt_r10': 93.7,
     'txt_r_mean': 85.43333333333334,
     'img_r1': 58.88,
     'img_r5': 83.82,
     'img_r10': 91.02,
     'img_r_mean': 77.90666666666665,
     'r_mean': 81.66999999999999
     }

]

# 不同模型的参数
model_params = [
    {'learning_rate': 0, 'batch_size': 0, 'epochs': 0},
    {'learning_rate': 0.0001, 'batch_size': 16, 'epochs': 4},
    {'learning_rate': 0.0005, 'batch_size': 32, 'epochs': 6},
    {'learning_rate': 0.0001, 'batch_size': 32, 'epochs': 6},
    {'learning_rate': 0.0001, 'batch_size': 128, 'epochs': 4},
    {'learning_rate': 0.0001, 'batch_size': 128, 'epochs': 6},
    {'learning_rate': 0.0001, 'batch_size': 128, 'epochs': 8}
]

# 提取不同模型的指标
metrics = list(model_results[0].keys())

plt.figure(figsize=(10, 6))

for i, metric in enumerate(metrics):
    values = [result[metric] for result in model_results]
    plt.plot(range(len(model_results)), values, marker='o', label=metric)

plt.xlabel('Models')
plt.ylabel('Values')
plt.title('Comparison of Evaluation Metrics Across Models')
plt.xticks(ticks=range(len(model_results)), labels=[str(params) for params in model_params], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
