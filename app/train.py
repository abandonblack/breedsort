from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from app.model import SUPPORTED_ARCHES, build_model

DATASET_CATALOG = {
    "oxford_iiit_pet": {
        "name": "Oxford-IIIT Pet Dataset",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "description": "37 个猫狗品种，官方标注质量高，适合品种分类。",
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练猫狗品种分类模型（ResNet34 / ResNet34-SE + Oxford-IIIT Pet）")
    parser.add_argument("--data-dir", type=Path, default=Path("data/oxford_iiit_pet"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--arches",
        nargs="+",
        default=["seresnet34", "resnet34"],
        choices=list(SUPPORTED_ARCHES),
        help="需要训练并对比的模型架构",
    )
    parser.add_argument("--no-download", action="store_true", help="不自动下载 Oxford-IIIT Pet")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否加载 ImageNet 预训练权重（推荐开启）",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            top1 = outputs.argmax(dim=1)
            correct_top1 += (top1 == labels).sum().item()

            topk = min(3, outputs.size(1))
            _, topk_idx = outputs.topk(k=topk, dim=1)
            correct_top3 += (topk_idx == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += images.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "top1": correct_top1 / max(total, 1),
        "top3": correct_top3 / max(total, 1),
    }


def _class_name_from_path(image_path: str) -> str:
    image_name = Path(image_path).stem
    return "_".join(image_name.split("_")[:-1]).lower()


def build_class_names(dataset: datasets.OxfordIIITPet) -> list[str]:
    id_to_name: dict[int, str] = {}
    for image_path, label in zip(dataset._images, dataset._labels):
        class_index = int(label)
        if class_index not in id_to_name:
            id_to_name[class_index] = _class_name_from_path(image_path)
    return [id_to_name[idx] for idx in sorted(id_to_name.keys())]


def make_splits(
    train_dataset: datasets.OxfordIIITPet,
    val_dataset: datasets.OxfordIIITPet,
    val_split: float,
    seed: int,
) -> tuple[Subset, Subset]:
    rng = random.Random(seed)
    class_buckets: dict[int, list[int]] = {}
    for idx, label in enumerate(train_dataset._labels):
        class_buckets.setdefault(int(label), []).append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in class_buckets.values():
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * val_split))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return Subset(train_dataset, train_indices), Subset(val_dataset, val_indices)


def train_one_model(
    arch: str,
    class_names: list[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model = build_model(num_classes=len(class_names), arch=arch, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = max(1, args.epochs // 10)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(args.epochs - warmup_epochs, 1),
                eta_min=args.lr * 0.05,
            ),
        ],
        milestones=[warmup_epochs],
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_acc = 0.0
    best_path = args.save_dir / f"breednet_{arch}_best.pth"

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

        train_loss = running_loss / max(len(train_loader.dataset), 1)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["top1"])

        print(
            f"[{arch}] Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['top1']:.4f} "
            f"| lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if val_metrics["top1"] > best_acc:
            best_acc = val_metrics["top1"]
            torch.save(
                {
                    "arch": arch,
                    "model_state_dict": model.state_dict(),
                    "classes": class_names,
                    "image_size": args.image_size,
                    "dataset_preset": "oxford_iiit_pet",
                    "best_val_acc": best_acc,
                },
                best_path,
            )

    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(
        f"[{arch}] Test(top1/top3): {test_metrics['top1']:.4f}/{test_metrics['top3']:.4f}"
    )

    return {
        "arch": arch,
        "best_val_acc": best_acc,
        "checkpoint": str(best_path),
        "history": history,
        "test": test_metrics,
    }


def plot_curves(results: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_path = output_dir / "loss_compare.png"
    acc_path = output_dir / "valacc_compare.png"

    plt.figure(figsize=(8, 5))
    for item in results:
        history = item["history"]
        plt.plot(history["epoch"], history["train_loss"], label=f"{item['arch']} train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for item in results:
        history = item["history"]
        plt.plot(history["epoch"], history["val_acc"], label=f"{item['arch']} val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=160)
    plt.close()

    return loss_path, acc_path


def plot_test_bar(results: list[dict[str, object]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    bar_path = output_dir / "test_topk_compare.png"

    arches = [item["arch"] for item in results]
    top1 = [item["test"]["top1"] for item in results]
    top3 = [item["test"]["top3"] for item in results]

    x = range(len(arches))
    width = 0.36

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], top1, width=width, label="Top1 Acc")
    plt.bar([i + width / 2 for i in x], top3, width=width, label="Top3 Acc")
    plt.xticks(list(x), arches)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Official Test Set: Top1 vs Top3")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160)
    plt.close()

    return bar_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_info = DATASET_CATALOG["oxford_iiit_pet"]
    print(f"使用数据集：{dataset_info['name']} | {dataset_info['url']}")

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
    eval_transform = transforms.Compose(
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
        transform=eval_transform,
        download=False,
    )
    test_set = datasets.OxfordIIITPet(
        root=str(args.data_dir),
        split="test",
        target_types="category",
        transform=eval_transform,
        download=False,
    )

    class_names = build_class_names(full_train)
    if len(class_names) < 2:
        raise ValueError("至少需要两个品种类别进行训练")

    train_set, val_set = make_splits(full_train, val_base, args.val_split, args.seed)

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
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for arch in args.arches:
        results.append(train_one_model(arch, class_names, train_loader, val_loader, test_loader, args, device))

    fig_dir = args.save_dir / "figures"
    loss_fig, acc_fig = plot_curves(results, fig_dir)
    topk_fig = plot_test_bar(results, fig_dir)

    metrics_path = args.save_dir / "experiment_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "dataset_preset": "oxford_iiit_pet",
                "dataset_info": dataset_info,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "pretrained": args.pretrained,
                "seed": args.seed,
                "models": results,
                "figures": {
                    "loss": str(loss_fig),
                    "val_acc": str(acc_fig),
                    "test_topk": str(topk_fig),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n实验完成，产物如下：")
    print(f"- 指标 JSON: {metrics_path}")
    print(f"- Loss 对比图: {loss_fig}")
    print(f"- ValAcc 对比图: {acc_fig}")
    print(f"- Test Top1/Top3 对比图: {topk_fig}")


if __name__ == "__main__":
    main()
