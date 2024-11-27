# pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader

# TorchVision domain specific libs
from torchvision import datasets # contains Dataset which is why we dont import it from torch.utils.data, also contains Zalando dataset
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#for X, y in test_dataloader:
#    print(f"Shape of X [N, C, H, W]: {X.shape}")
#    print(f"Shape of y: {y.shape} {y.dtype}")
#    break


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



# Model definition

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------")
    train(train_dataloader, model, loss, optimizer)
    test(test_dataloader, model, loss)
print("Done")

model_filename = "zalando.pth"

torch.save(model.state_dict(), model_filename)
print("Saved PyTorch Model State to ", model_filename)


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# lets test a subset of test data
subset_size = 100
subset_loss, subset_correct = 0, 0

model.eval()

with torch.no_grad():
    for i in range(subset_size):
        x, y = test_data[i][0].to(device), test_data[i][1]
        x = x.unsqueeze(0)

        pred = model(x)
        predicted_label = pred.argmax(1).item()
        subset_loss += loss(pred, torch.tensor([y], device=device)).item()
        subset_correct += 1 if predicted_label == y else 0
        print(f"Sample {i+1}: Predicted: {classes[predicted_label]}, Actual: {classes[y]}")

average_loss = subset_loss / subset_size
average_accuracy = (subset_correct / subset_size) * 100
print(f"\nAverage Loss: {average_loss:.6f}")
print(f"Average Accuracy: {average_accuracy:.2f}%")


