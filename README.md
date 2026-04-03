# 猫狗品种精准识别平台（Oxford-IIIT + ResNet 版）

一个基于 **PyTorch + FastAPI** 的 Web 应用，支持：

- 图片上传并识别猫狗品种
- 返回 Top3 候选与置信度
- 用户反馈提交与展示

## 模型与数据集约束（已精简）

本项目已精简为：

- **仅保留残差网络 `ResBreedNet`**（移除了轻量 CNN）
- **仅使用 Oxford-IIIT Pet Dataset**（通过 `torchvision.datasets.OxfordIIITPet` 直接加载）

你也可以通过接口查看数据集信息：`GET /api/datasets`。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 训练模型（自动下载 Oxford-IIIT）

```bash
python -m app.train --data-dir data/oxford_iiit_pet --epochs 20
```

说明：

- 默认会自动下载官方数据集到 `--data-dir`
- 若你已手动下载完成，可加 `--no-download`

训练后输出：

- `artifacts/breednet.pth`（权重）
- `artifacts/breednet.json`（训练信息）

## 3. 启动服务

```bash
python app/main.py
```

访问：`http://127.0.0.1:8000`

## 接口

- `GET /api/datasets`：查看数据集清单（当前仅 Oxford-IIIT）
- `POST /api/predict`：上传图片字段 `image`
- `POST /api/feedback`：字段 `nickname`、`message`、`rating`
- `GET /api/feedback`：查看最近反馈
