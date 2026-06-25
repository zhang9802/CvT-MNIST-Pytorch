import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from torchsummary import summary
from torchvision import datasets
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from models import CvT
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
batch_size = 64
lr = 1e-4
epochs = 10
in_channel, out_channel = [1, 64, 128], [64, 128, 256]
kernel_size, stride = [12, 4, 2], [10, 3, 1]
n_class =  10
model = CvT(in_channel, out_channel,n_class,kernel_size, stride).to(device)


mnist_train = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
mnist_test = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, drop_last=False)

# model = nn.Sequential(
#     nn.Dropout(0.2),
#     nn.Linear(28*28, 128),
#     nn.ReLU(),
#     nn.Dropout(0.2),
#     nn.Linear(128, 10)
# ).to(device)

# summary(model, input_size=(28, 28))
# print(model(torch.randn(10, 224,224).to(device)).shape)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_all = []
for epoch in range(epochs):
    loss_epoch = 0
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss {loss.item():.4f}')
    loss_all.append(loss_epoch / len(train_loader))

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

plt.figure()
plt.plot(loss_all, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig('Loss.png',dpi=900)
