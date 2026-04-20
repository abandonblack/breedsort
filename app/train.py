from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from app.model import build_model

DATASET_CATALOG = {
    "oxford_iiit_pet": {
        "name": "Oxford-IIIT Pet Dataset",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "description": "37 个猫狗品种，官方标注质量高，适合品种分类。",
    },
    "imagefolder_extra": {
        "name": "自定义 ImageFolder 额外数据集",
        "url": "local_path_only",
        "description": "用于泛化性评估。目录结构需为 root/class_x/*.jpg。",
    },
}


class RemappedImageFolder(Dataset):
    """将 ImageFolder 类别重映射到训练类别索引，支持泛化评估。"""

    def __init__(self, root: Path, transform: transforms.Compose, class_to_idx: dict[str, int]) -> None:
        super().__init__()
        base = datasets.ImageFolder(root=str(root), transform=transform)
        self.samples: list[tuple[str, int]] = []
        self.transform = transform

        dataset_classes = {name.lower(): idx for name, idx in base.class_to_idx.items()}
        missing = sorted(set(class_to_idx.keys()) - set(dataset_classes.keys()))
        if missing:
            raise ValueError(
                "额外数据集缺少训练类别，无法直接评估。"
                f"缺失类别数量={len(missing)}，例如: {missing[:5]}"
            )

        for path, folder_idx in base.samples:
            class_name = base.classes[folder_idx].lower()
            if class_name in class_to_idx:
                self.samples.append((path, class_to_idx[class_name]))

        if not self.samples:
            raise ValueError("额外数据集没有可用样本，请检查类别目录命名是否与训练类别一致。")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = datasets.folder.default_loader(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练猫狗品种分类模型（支持 ResNet34/ResNet34-SE、F1、泛化评估与曲线图）"
    )
    parser.add_argument("--arch", type=str, default="seresnet34", choices=["seresnet34", "resnet34"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/oxford_iiit_pet"))
    parser.add_argument("--extra-data-dir", type=Path, default=None, help="用于泛化评估的 ImageFolder 数据集路径")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/breednet.pth"))
    parser.add_argument("--no-download", action="store_true", help="已下载时关闭自动下载")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def compute_macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    f1_sum = 0.0
    for cls in range(num_classes):
        pred_pos = preds == cls
        label_pos = labels == cls
        tp = (pred_pos & label_pos).sum().item()
        fp = (pred_pos & ~label_pos).sum().item()
        fn = (~pred_pos & label_pos).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_sum += f1
    return f1_sum / max(num_classes, 1)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    if total == 0:
        return 0.0, 0.0, 0.0

    merged_preds = torch.cat(all_preds)
    merged_labels = torch.cat(all_labels)
    macro_f1 = compute_macro_f1(merged_preds, merged_labels, num_classes=num_classes)
    return total_loss / total, correct / total, macro_f1


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


def plot_history(history: dict[str, list[float]], output_path: Path, arch: str) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0, 0].plot(epochs, history["val_loss"], label="val_loss")
    if "extra_loss" in history:
        axes[0, 0].plot(epochs, history["extra_loss"], label="extra_loss")
    axes[0, 0].set_title("Loss vs Epoch")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, history["val_acc"], label="val_acc")
    if "extra_acc" in history:
        axes[0, 1].plot(epochs, history["extra_acc"], label="extra_acc")
    axes[0, 1].set_title("Accuracy vs Epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, history["val_f1"], label="val_f1")
    if "extra_f1" in history:
        axes[1, 0].plot(epochs, history["extra_f1"], label="extra_f1")
    axes[1, 0].set_title("Macro-F1 vs Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epochs, history["lr"], label="learning_rate")
    axes[1, 1].set_title("Learning Rate vs Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Training Dynamics ({arch})")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_info = DATASET_CATALOG["oxford_iiit_pet"]
    print(f"使用数据集：{dataset_info['name']} | {dataset_info['url']}")
    if args.extra_data_dir is not None:
        print(f"额外泛化评估数据集：{args.extra_data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_train = datasets.OxfordIIITPet(
        root=str(args.data_dir),
        split="trainval",
        target_types="category",
        transform=train_transform,
        download=not args.no_download,
    )
    val_base = datasets.OxfordIIITPet(
        root=str(args.data_dir),
        split="trainval",
        target_types="category",
        transform=val_transform,
        download=False,
    )
    class_names = build_class_names(full_train)

    if len(class_names) < 2:
        raise ValueError("至少需要两个品种类别进行训练")

    val_size = int(len(full_train) * args.val_split)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, _ = random_split(full_train, [train_size, val_size], generator=generator)
    _, val_set = random_split(val_base, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    extra_loader = None
    extra_info = None
    if args.extra_data_dir is not None:
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        extra_dataset = RemappedImageFolder(args.extra_data_dir, val_transform, class_to_idx=class_to_idx)
        extra_loader = DataLoader(
            extra_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        extra_info = {
            "name": DATASET_CATALOG["imagefolder_extra"]["name"],
            "path": str(args.extra_data_dir),
            "num_samples": len(extra_dataset),
        }

    model = build_model(num_classes=len(class_names), arch=args.arch).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
    }
    if extra_loader is not None:
        history["extra_loss"] = []
        history["extra_acc"] = []
        history["extra_f1"] = []

    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(len(train_set), 1)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, num_classes=len(class_names))

        message = (
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if extra_loader is not None:
            extra_loss, extra_acc, extra_f1 = evaluate(
                model,
                extra_loader,
                criterion,
                device,
                num_classes=len(class_names),
            )
            history["extra_loss"].append(extra_loss)
            history["extra_acc"].append(extra_acc)
            history["extra_f1"].append(extra_f1)
            message += f" | extra_acc={extra_acc:.4f} | extra_f1={extra_f1:.4f}"

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        history["lr"].append(current_lr)
        message += f" | lr={current_lr:.6f}"
        print(message)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "arch": args.arch,
                    "model_state_dict": model.state_dict(),
                    "classes": class_names,
                    "image_size": args.image_size,
                    "dataset_preset": "oxford_iiit_pet",
                },
                args.save_path,
            )

    meta_path = args.save_path.with_suffix(".json")
    curves_path = args.save_path.with_name(f"{args.save_path.stem}_curves.png")
    history_path = args.save_path.with_name(f"{args.save_path.stem}_history.json")

    meta_path.write_text(
        json.dumps(
            {
                "arch": args.arch,
                "dataset_preset": "oxford_iiit_pet",
                "dataset_info": dataset_info,
                "extra_dataset_info": extra_info,
                "classes": class_names,
                "best_val_acc": best_acc,
                "best_val_f1": max(history["val_f1"]) if history["val_f1"] else 0.0,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "seed": args.seed,
                "history_path": str(history_path),
                "curves_path": str(curves_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_history(history, curves_path, arch=args.arch)

    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")
    print(f"权重已保存: {args.save_path}")
    print(f"训练动态曲线已保存: {curves_path}")


if __name__ == "__main__":
    main()
