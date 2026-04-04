# 猫狗品种精准识别平台（Oxford-IIIT + 手写 ResNet34-SE）

一个基于 **PyTorch + FastAPI** 的 Web 应用，支持：

- 图片上传并识别猫狗品种
- 返回 Top3 候选与置信度
- 用户反馈提交与展示

## 模型与数据集约束

本项目当前使用：

- **手写 ResNet34-SE**（含 SE 通道注意力，不调用 `torchvision.models`）
- **仅使用 Oxford-IIIT Pet Dataset**（通过 `torchvision.datasets.OxfordIIITPet` 直接加载）

你也可以通过接口查看数据集信息：`GET /api/datasets`。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 训练模型（自动下载 Oxford-IIIT）

```bash
python -m app.train --data-dir data/oxford_iiit_pet --epochs 40 --batch-size 32
```

可选高频参数：

- `--lr`：初始学习率（默认 `3e-4`）
- `--weight-decay`：权重衰减（默认 `1e-4`）
- `--label-smoothing`：标签平滑（默认 `0.1`）
- `--workers`：DataLoader 线程数（默认 `4`）
- `--seed`：随机种子（默认 `42`）
- `--no-download`：已下载数据时关闭自动下载

训练后输出：

- `artifacts/breednet.pth`（权重）
- `artifacts/breednet.json`（训练信息）

## 3. 启动服务

```bash
python -m app.main
```

访问：`http://127.0.0.1:8000`

## 接口

- `GET /api/datasets`：查看数据集清单（当前仅 Oxford-IIIT）
- `POST /api/predict`：上传图片字段 `image`
- `POST /api/feedback`：字段 `nickname`、`message`、`rating`
- `GET /api/feedback`：查看最近反馈
