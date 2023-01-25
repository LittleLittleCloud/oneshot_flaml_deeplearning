from model import Model
from torchvision import datasets
import torch
from torch.utils.data import DataLoader
import torchvision

def train_model(
    dataset: datasets.ImageFolder,
    batch_size: int,
    learning_rate: float,
    graident: float,
    square: float,
    eps: float,
    weight_decay:float,
    num_classes: int,
    num_epochs: int,
    device: torch.device):
    model = Model(num_of_class=num_classes, device=device)
    optimizer = torch.optim.Adam(model.parameters, lr=learning_rate, betas=(graident, square), eps=eps, weight_decay=weight_decay)
    cross_entropy = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        model.train()

        train_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model.forward(inputs)
            loss = cross_entropy.forward(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_data_loader)
        epoch_acc = running_corrects.double() / len(train_data_loader)

        print('train - epoch: {} loss: {:.4f}, acc: {:.4f}'.format(epoch,
                                                        epoch_loss,
                                                        epoch_acc))


