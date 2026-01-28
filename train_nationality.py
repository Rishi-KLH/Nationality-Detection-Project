import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_CSV = "FairFace/train_labels.csv"
VAL_CSV   = "FairFace/val_labels.csv"
TRAIN_DIR = "FairFace/train"
VAL_DIR   = "FairFace/val"

print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("VAL_CSV exists:", os.path.exists(VAL_CSV))
print("TRAIN_DIR exists:", os.path.exists(TRAIN_DIR))
print("VAL_DIR exists:", os.path.exists(VAL_DIR))

BATCH_SIZE = 64
EPOCHS = 1
LR = 0.0003

class FairFaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.race_classes = sorted(self.df["race"].unique())
        self.race_to_idx = {r:i for i, r in enumerate(self.race_classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filename = row["file"].replace("train/", "").replace("val/", "")
        img_path = os.path.join(self.img_dir, filename)

        img = Image.open(img_path).convert("RGB")
        label = self.race_to_idx[row["race"]]

        if self.transform:
            img = self.transform(img)

        return img, label

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_data = FairFaceDataset(TRAIN_CSV, TRAIN_DIR, transform)
val_data   = FairFaceDataset(VAL_CSV, VAL_DIR, transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

num_classes = len(train_data.race_classes)
print("Race classes:", train_data.race_classes)

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct/total

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # ✅ print every 50 batches
        if i % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

    val_acc = evaluate(val_loader)
    print(f"✅ Epoch {epoch+1}/{EPOCHS} Completed | Avg Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

torch.save({
    "model_state": model.state_dict(),
    "classes": train_data.race_classes
}, "nationality_race_model.pth")

print("✅ Model saved as nationality_race_model.pth")