import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# code will run in gpu if available and if the flag is set to True, else it will run on cpu
use_gpu = True
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # hidden cov(1)
        t = t
        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)        
        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)
        t = self.out(t)
        return t


train_data = torchvision.datasets.FashionMNIST(
    '/home/mostafa/work/learn Pytorch/',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]))


train_loader = torch.utils.data.DataLoader(train_data, batch_size=100)
total_loss = 0
total_correct = 0
network = Network()
optimizer = optim.Adam(network.parameters(), lr=0.01)

for batch in train_loader:
    images, labels = batch
    preds = network(images)
    loss = F.cross_entropy(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    loss.item()
    optimizer.step()
    total_loss += loss.item()
    total_correct += get_num_correct(preds, labels)
print("Loss", total_loss, "Total coreect =",total_correct)
print(total_correct / len(train_data))