import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "nationality_race_model.pth"

# Load saved model
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
classes = checkpoint["classes"]

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model_state"])
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_race(img_np):
    img = Image.fromarray(img_np).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)     # ✅ confidence scores
        pred_idx = probs.argmax(1).item()
        confidence = probs[0][pred_idx].item()

    race = classes[pred_idx]
    return race, confidence