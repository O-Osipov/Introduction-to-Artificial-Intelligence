import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ---------------------------------------------------------
# Глобальные константы
# ---------------------------------------------------------
IMG_SIZE = 64

# ---------------------------------------------------------
# 1. Функция обработки изображения (на верхнем уровне!)
# ---------------------------------------------------------
def process_image(args):
    """Обработка одного изображения"""
    img_path, label = args
    try:
        img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)
        return {'label': label, 'path': img_path, 'pixels': img_array}
    except:
        return None

# ---------------------------------------------------------
# 2. Определение Сверточной Нейронной Сети (CNN)
# ---------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------------------------------------
# 3. Загрузка датасета
# ---------------------------------------------------------
def load_simpsons_dataset(dataset_path='./archive/simpsons_dataset', img_size=64):
    """Загрузка и предобработка датасета Симпсонов"""
    global IMG_SIZE
    IMG_SIZE = img_size
    
    print("Загрузка данных...")
    
    if not os.path.exists(dataset_path):
        print(f"Ошибка: Путь '{dataset_path}' не найден.")
        return None, None, None

    # Собираем все пути заранее
    tasks = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    tasks.append((os.path.join(label_path, img_name), label))

    print(f"Всего изображений: {len(tasks)}")
    print(f"Ядер процессора: {cpu_count()}")

    # Запускаем параллельно
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_image, tasks), total=len(tasks), desc="Обработка"))

    # Фильтруем ошибки и создаем DataFrame
    df = pd.DataFrame([r for r in results if r is not None])
    print(f"\nГотово! Обработано: {len(df)} изображений")
    
    if len(df) == 0:
        return None, None, None

    # Кодирование меток (строка -> число)
    labels_unique = df['label'].unique()
    label_to_idx = {label: idx for idx, label in enumerate(labels_unique)}
    df['label_idx'] = df['label'].map(label_to_idx)
    
    print(f"Количество классов: {len(labels_unique)}")
    print(f"Классы: {labels_unique}")
    
    return df, label_to_idx, len(labels_unique)

# ---------------------------------------------------------
# 4. Кастомный Dataset с поддержкой трансформов
# ---------------------------------------------------------
class SimpsonsDataset(Dataset):
    """Кастомный датасет с применением трансформов"""
    def __init__(self, X, Y, transforms=None):
        self.X = X  # numpy array (N, H, W, C)
        self.Y = Y  # numpy array (N,)
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.Y[idx]
        
        # Конвертируем numpy array в PIL Image для transforms
        img = Image.fromarray((img * 255).astype(np.uint8))
        
        # Применяем трансформы (аугментацию)
        if self.transforms:
            img = self.transforms(img)
        
        return img, label

# ---------------------------------------------------------
# 5. Основная функция
# ---------------------------------------------------------
def main():
    # Проверка доступности GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Загрузка данных
    df, label_to_idx, num_classes = load_simpsons_dataset()
    if df is None:
        return

    # Подготовка данных
    X = np.stack(df['pixels'].values).astype(np.float32) / 255.0
    Y = df['label_idx'].values.astype(np.int64)

    print(f"Всего samples: {X.shape[0]}, Форма: {X.shape}")

    # -----------------------------------------------------
    # ДОБАВЛЕНО: Аугментация данных
    # -----------------------------------------------------
    # Трансформы для обучения (с аугментацией)
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),      # Случайный горизонтальный переворот
        transforms.RandomRotation(15),               # Случайный поворот до 15 градусов
        transforms.ColorJitter(brightness=0.2,       # Изменение яркости
                               contrast=0.2,         # Изменение контраста
                               saturation=0.2),      # Изменение насыщенности
        transforms.ToTensor(),                       # Конвертация в тензор [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Нормализация (ImageNet stats)
                             std=[0.229, 0.224, 0.225])
    ])

    # Трансформы для тестирования (без аугментации)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # -----------------------------------------------------

    # Разделение на Train/Test (80/20)
    train_size = int(0.8 * len(X))
    test_size = len(X) - train_size
    
    indices = torch.randperm(len(X)).tolist()
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]

    # Создание датасетов с трансформами
    train_dataset = SimpsonsDataset(X_train, Y_train, transforms=train_transforms)
    test_dataset = SimpsonsDataset(X_test, Y_test, transforms=test_transforms)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Инициализация модели
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    epochs = 10
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    print(f"{'Epoch': <8} {'Train Loss': <15} {'Train Acc': <12} {'Test Loss': <15} {'Test Acc': <12}")
    print("-" * 65)

    # Цикл обучения
    for epoch in range(epochs):
        # Обучение
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Тестирование
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_acc = correct / total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"{epoch+1: <8} {train_loss: <15.4f} {train_acc: <12.4f} {test_loss: <15.4f} {test_acc: <12.4f}")

    # Построение графиков
    epochs_range = range(1, epochs + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(epochs_range, train_accuracies, label='Train Acc', marker='o')
    ax1.plot(epochs_range, test_accuracies, label='Test Acc', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    ax2.plot(epochs_range, test_losses, label='Test Loss', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_plot_augmented.png', dpi=150)
    plt.show()
    print("\nГрафик сохранён в 'training_plot_augmented.png'")

if __name__ == "__main__":
    main()