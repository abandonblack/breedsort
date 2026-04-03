from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from app.model import build_model

DATASET_CATALOG = {
    "oxford_iiit_pet": {
        "name": "Oxford-IIIT Pet Dataset",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "description": "37 个猫狗品种，官方标注质量高，适合品种分类。",
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练猫狗品种分类模型（仅 ResNet + Oxford-IIIT Pet）")
    parser.add_argument("--data-dir", type=Path, default=Path("data/oxford_iiit_pet"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/breednet.pth"))
    parser.add_argument("--no-download", action="store_true", help="不自动下载 Oxford-IIIT Pet")
    return parser.parse_args()


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def _class_name_from_path(image_path: str) -> str:
    image_name = Path(image_path).stem
    return "_".join(image_name.split("_")[:-1]).lower()


def build_class_names(dataset: datasets.OxfordIIITPet) -> list[str]:
    id_to_name: dict[int, str] = {}
    for image_path, label in zip(dataset._images, dataset._labels):
        class_index = int(label) - 1
        if class_index not in id_to_name:
            id_to_name[class_index] = _class_name_from_path(image_path)
    return [id_to_name[idx] for idx in sorted(id_to_name.keys())]


def main() -> None:
    args = parse_args()

    dataset_info = DATASET_CATALOG["oxford_iiit_pet"]
    print(f"使用数据集：{dataset_info['name']} | {dataset_info['url']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.OxfordIIITPet(
        root=str(args.data_dir),
        split="trainval",
        target_types="category",
        transform=transform,
        download=not args.no_download,
    )
    class_names = build_class_names(dataset)

    if len(class_names) < 2:
        raise ValueError("至少需要两个品种类别进行训练")

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(len(train_set), 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "arch": "resbreednet",
                    "model_state_dict": model.state_dict(),
                    "classes": class_names,
                    "image_size": args.image_size,
                    "dataset_preset": "oxford_iiit_pet",
                },
                args.save_path,
            )

    meta_path = args.save_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "arch": "resbreednet",
                "dataset_preset": "oxford_iiit_pet",
                "dataset_info": dataset_info,
                "classes": class_names,
                "best_val_acc": best_acc,
                "epochs": args.epochs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")
    print(f"权重已保存: {args.save_path}")


if __name__ == "__main__":
    main()
