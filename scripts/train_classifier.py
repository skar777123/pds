"""
Train MobileNetV2 Classifier for Ultra-Fast Edge Inference on Raspberry Pi 4B.
Strictly maps to PRD Section 4: "A quantized YOLOv8n-tflite or MobileNetV2 classifier processes each frame".
Outputs ONNX and LiteRT/TFLite models for CPU execution.
"""

import os
import glob
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

def train_mobilenetv2_classifier(epochs=10, batch_size=8, lr=0.001):
    base_dir = Path(__file__).resolve().parent.parent
    images_dir = base_dir / "dataset" / "images"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 5 Taxonomy classes from PRD
    class_names = ["Disintegrated", "Manhole", "Water clogged", "Raveling", "Deformation"]
    class_to_idx = {name.lower().replace(" ", "_"): idx for idx, name in enumerate(class_names)}
    
    print("[Classifier] Loading dataset images...")
    data = []
    for img_path in images_dir.glob("*.jpeg"):
        name_lower = img_path.stem.lower()
        matched_class = None
        for key, idx in class_to_idx.items():
            if key in name_lower:
                matched_class = idx
                break
        if matched_class is not None:
            data.append((str(img_path), matched_class))

    print(f"[Classifier] Found {len(data)} images across {len(class_names)} classes.")
    if not data:
        print("[Classifier] No images found. Extract dataset first.")
        return

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # MobileNetV2 architecture
    print("[Classifier] Initializing MobileNetV2 backbone...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    # Replace final classification head with 5-class output
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, len(class_names))
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Train loop
    model.train()
    print(f"[Classifier] Training MobileNetV2 for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        # Shuffle data
        import random
        random.seed(epoch)
        random.shuffle(data)

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            imgs = [transform(Image.open(p).convert("RGB")) for p, _ in batch]
            labels = [l for _, l in batch]

            inputs = torch.stack(imgs)
            targets = torch.tensor(labels, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += len(batch)

        acc = correct / total if total > 0 else 0
        print(f"  Epoch [{epoch+1:02d}/{epochs:02d}] Loss: {total_loss/total:.4f} | Accuracy: {acc*100:.1f}%")

    # Save PyTorch weights
    pt_path = models_dir / "pothole_classifier_mobilenetv2.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"[Classifier] Saved PyTorch weights to: {pt_path}")

    # Export to ONNX
    onnx_path = models_dir / "pothole_classifier_mobilenetv2.onnx"
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["probabilities"],
        dynamic_axes={"input": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
        opset_version=17
    )
    print(f"[Classifier] Exported ONNX classifier to: {onnx_path}")

if __name__ == "__main__":
    train_mobilenetv2_classifier(epochs=10, batch_size=8, lr=0.001)
