import torch
from PIL import Image
import torchvision.transforms as transforms
from model import get_model

# --- 修改成你的圖片名稱 ---
image_path = 'test.jpg'
model_path = 'model_epoch_6.pth'  # 改成你最好的模型檔名

# --- 影像前處理 ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# --- 載入模型 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(num_classes=8, pretrained=False).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- 載入圖片並預測 ---
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

# --- 對應表情標籤 ---
label_map = {
    0: 'Neutral',
    1: 'Happiness',
    2: 'Sadness',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger',
    7: 'Contempt'
}

print(f"預測結果是：{label_map[int(predicted)]}")
