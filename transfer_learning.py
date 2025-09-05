import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import multiprocessing
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    multiprocessing.freeze_support() # 윈도우 환경에서 필요
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = '.\\archive\\Bone Break Classification\\Bone Break Classification'
    train_dir = '.\\archive\\Bone Break Classification\\Bone Break Classification\\Train'
    val_dir = '.\\archive\\Bone Break Classification\\Bone Break Classification\\Test'

    # 이미지 크기 및 정규화 값 설정 (ResNet 모델에 맞춰 조정)
    image_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 학습 데이터셋 변환
    train_transforms = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 검증 데이터셋 변환
    val_transforms = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 학습 데이터셋 로드
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transforms
    )

    # 검증 데이터셋 로드
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transforms
    )
    print('train_dataset: ', len(train_dataset))
    print('val_dataset: ', len(val_dataset))

    # 라벨링
    labels_map = {
        0: 'Avulsion fracture',
        1: 'Comminuted fracture',
        2: 'Fracture Dislocation',
        3: 'Greenstick fracture',
        4: 'Hairline Fracture',
        5: 'Impacted fracture',
        6: 'Longitudinal fracture',
        7: 'Oblique fracture',
        8: 'Pathological fracture',
        9: 'Spiral Fracture'
    }

    # DataLoader 정의
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 클래스 이름 및 개수 확인
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print("Class names:", class_names)
    print("Number of classes:", num_classes)

    #  사전 학습된 ResNet 모델 로드 및 수정
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft = model_ft.to(device)

    #  손실 함수 및 Optimizer 설정
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    #  모델 학습 함수 정의
    def train_model(model, criterion, optimizer, scheduler, num_epochs):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print('-' * 10)

            model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects.double() / len(train_dataset)
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc.item())
            print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            model.eval()
            val_loss = 0.0
            val_corrects = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)

            epoch_val_loss = val_loss / len(val_dataset)
            epoch_val_acc = val_corrects.double() / len(val_dataset)
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc.item())
            print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

            if scheduler:
                scheduler.step()

        return model, train_losses, val_losses, train_accuracies, val_accuracies

    """#평가하려면 여기서부터 주석
    #  모델 학습 실행
    num_epochs = 100
    trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(model_ft, criterion,
                                                                                            optimizer_ft, scheduler,
                                                                                            num_epochs)

    #  학습 결과 시각화
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    
    # 모델 저장
    torch.save(trained_model.state_dict(), '모델별 테스트 결과/resnet18_finetuned/resnet18_finetuned.pth')
    #평가하려면 여기까지 주석"""

    # 모델 로드
    #Path 변수에 각 모델별 파라미터 경로 넣어서 결과 확인할 수 있음
    PATH = '모델별 테스트 결과/resnet18_finetuned/resnet18_finetuned.pth'

    net = model_ft
    net.load_state_dict(torch.load(PATH))

    # 모델 평가
    def evaluate_model(model, dataloader):
        model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predictions == labels).sum().item()
        accuracy = total_correct / total_samples
        print(f"Evaluation Accuracy: {accuracy:.4f}")

    # 모델 테스트
    def imshow(image):
        image = image.cpu().clone().detach()
        image = image * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1) # 정규화 되돌리기
        image = image.numpy()
        plt.imshow(np.transpose(image, (1, 2, 0)))
        plt.show()

    dataiter = iter(val_loader)
    images, label = dataiter.__next__()
    images = images.to(device)  # 입력 데이터 GPU 이동
    imshow(torchvision.utils.make_grid((images[:6].cpu())))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # 6개 이미지와 예측 결과 표시
    figure = plt.figure(figsize=(16, 8))
    for i in range(6):
        ax = figure.add_subplot(2, 3, i + 1)
        image = images[i].cpu().clone().detach()
        image = image * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)  # 정규화 되돌리기
        true_label = labels_map[label[i].item()]
        pred_label = labels_map[predicted[i].item()]
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.set_title(f'True: {true_label}\nPredicted: {pred_label}')
        ax.axis('off')
    plt.show()

    evaluate_model(net,val_loader)
