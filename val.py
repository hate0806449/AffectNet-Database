import torch
from torch.utils.data import DataLoader
from dataset import AffectNetDataset
from model import get_model
import torchvision.transforms as transforms
import torch.nn as nn

def validate():
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    val_dataset = AffectNetDataset('val_set/images', 'val_set/annotations', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=8, pretrained=False).to(device)
    model.load_state_dict(torch.load('model_epoch_6.pth'))  # 載入訓練後模型
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Loss: {total_loss/total:.4f}, Accuracy: {correct/total:.4f}')

if __name__ == '__main__':
    validate()
