# 猫狗品种精准识别平台（Oxford-IIIT + 手写 ResNet34 / ResNet34-SE）

一个基于 **PyTorch + FastAPI** 的 Web 应用，支持：

- 图片上传并识别猫狗品种
- 返回 Top3 候选与置信度
- 用户反馈提交与展示
- 训练时记录 Loss/Acc/F1/LR 动态并自动绘图

## 模型与数据集

当前支持两种手写模型（不调用 `torchvision.models`）：

- **ResNet34-SE**（带 SE 通道注意力）
- **ResNet34**（无注意力机制基线）

训练主数据集：

- **Oxford-IIIT Pet Dataset**（`torchvision.datasets.OxfordIIITPet`）

可选泛化评估数据集：

- **自定义 ImageFolder**（目录结构：`root/class_x/*.jpg`）
- 类名需与 Oxford-IIIT 的类别名一致（小写下划线形式），用于评估跨域泛化表现

你可以通过接口查看数据集信息：`GET /api/datasets`。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 训练模型（自动下载 Oxford-IIIT）

### 2.1 训练带注意力模型（ResNet34-SE）

```bash
python -m app.train --arch seresnet34 --data-dir data/oxford_iiit_pet --epochs 40 --batch-size 32
```

### 2.2 训练无注意力基线（ResNet34）

```bash
python -m app.train --arch resnet34 --data-dir data/oxford_iiit_pet --epochs 40 --batch-size 32 --save-path artifacts/resnet34_baseline.pth
```

### 2.3 加入额外数据集进行泛化评估

```bash
python -m app.train \
  --arch seresnet34 \
  --data-dir data/oxford_iiit_pet \
  --extra-data-dir data/generalization_set \
  --epochs 40
```

可选高频参数：

- `--lr`：初始学习率（默认 `3e-4`）
- `--weight-decay`：权重衰减（默认 `1e-4`）
- `--label-smoothing`：标签平滑（默认 `0.1`）
- `--workers`：DataLoader 线程数（默认 `4`）
- `--seed`：随机种子（默认 `42`）
- `--no-download`：已下载数据时关闭自动下载

训练后输出：

- `*.pth`：模型权重
- `*.json`：训练配置与最佳指标（含 best val acc / best val f1）
- `*_history.json`：按 epoch 的动态指标（train_loss/val_loss/val_acc/val_f1/lr/extra_*）
- `*_curves.png`：自动绘制的训练动态曲线图（Loss / Accuracy / Macro-F1 / LR）

## 3. 本科论文建议实验流程

1. 固定同一随机种子与超参数，分别训练：
   - `--arch resnet34`
   - `--arch seresnet34`
2. 对比两个实验输出的 `*_history.json` 与 `*_curves.png`。
3. 重点分析：
   - Val Accuracy 与 Macro-F1 差异
   - 收敛速度（前 10~20 epoch 曲线斜率）
   - 是否过拟合（train/val loss 分叉）
   - 在 `--extra-data-dir` 的泛化表现差异

## 4. 启动服务

```bash
python -m app.main
```

访问：`http://127.0.0.1:8000`

## 接口

- `GET /api/datasets`：查看数据集清单
- `POST /api/predict`：上传图片字段 `image`
- `POST /api/feedback`：字段 `nickname`、`message`、`rating`
- `GET /api/feedback`：查看最近反馈
