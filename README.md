# 猫狗品种精准识别平台（Oxford-IIIT + ResNet34 对比实验）

一个基于 **PyTorch + FastAPI** 的 Web 应用，支持：

- 图片上传并识别猫狗品种
- 返回 Top3 候选与置信度
- 用户反馈提交与展示
- 两种模型（ResNet34、ResNet34-SE）实验对比

## 模型与数据集约束

本项目当前使用：

- **手写 ResNet34-SE**（含 SE 通道注意力，不调用 `torchvision.models`）
- **手写 ResNet34**（无注意力机制）
- **仅使用 Oxford-IIIT Pet Dataset**（通过 `torchvision.datasets.OxfordIIITPet` 直接加载）

你也可以通过接口查看数据集信息：`GET /api/datasets`。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 训练与对比实验（默认 epoch=50）

```bash
python -m app.train --data-dir data/oxford_iiit_pet --epochs 50 --batch-size 32
```

默认会顺序训练两个模型：

- `seresnet34`（有注意力）
- `resnet34`（无注意力）

可选高频参数：

- `--arches`：选择训练模型，例如 `--arches seresnet34 resnet34`
- `--lr`：初始学习率（默认 `3e-4`）
- `--weight-decay`：权重衰减（默认 `1e-4`）
- `--label-smoothing`：标签平滑（默认 `0.1`）
- `--workers`：DataLoader 线程数（默认 `4`）
- `--seed`：随机种子（默认 `42`）
- `--no-download`：已下载数据时关闭自动下载

训练后输出：

- `artifacts/breednet_seresnet34_best.pth`
- `artifacts/breednet_resnet34_best.pth`
- `artifacts/experiment_metrics.json`
- `artifacts/figures/loss_compare.png`（两个模型 loss 曲线对比）
- `artifacts/figures/valacc_compare.png`（两个模型 val acc 曲线对比）
- `artifacts/figures/test_topk_compare.png`（官方 test 上 Top1/Top3 柱状图）

其中 test 指标来自 **Oxford-IIIT Pet 官方 `split="test"`**。

## 3. 启动服务

```bash
python -m app.main
```

访问：`http://127.0.0.1:8000`

> Web 服务默认读取 `artifacts/breednet.pth`。如果要用新的对比实验模型进行推理，请把对应权重重命名/拷贝为该路径，或自行扩展加载逻辑。

## 接口

- `GET /api/datasets`：查看数据集清单（当前仅 Oxford-IIIT）
- `POST /api/predict`：上传图片字段 `image`
- `POST /api/feedback`：字段 `nickname`、`message`、`rating`
- `GET /api/feedback`：查看最近反馈
