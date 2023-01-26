from src.model import Model
from src.dataset import *
from torchvision import datasets
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def train_and_evaluate_model(
    train_dataset,
    validate_dataset,
    num_classes: int,
    num_epochs: int,
    device: torch.device,
    batch_size = 128,
    learning_rate = 1e-3,
    graident = 0.9,
    square = 0.99,
    eps = 1e-8,
    weight_decay = 0):
    model = Model(num_of_class=num_classes, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(graident, square), eps=eps, weight_decay=weight_decay)
    cross_entropy = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        model.train()
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        running_loss = 0.0
        running_corrects = 0
        batch = 0
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
            correctness = torch.sum(preds == labels.data)
            print(f'batch {batch} loss {loss.item()} correctness {correctness * 1.0 / inputs.size(0)}')
            running_corrects += correctness
            batch += 1
        epoch_loss = running_loss / len(train_data_loader)
        epoch_correctness = running_corrects * 1.0 / len(train_data_loader)
        print('train - epoch: {} loss: {:.4f} correctness: {}'.format(epoch,
                                                        epoch_loss,
                                                        epoch_correctness))

    preds = []
    labels = []
    # using auc
    model.eval()
    validate_dataset = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
    for inputs, label in validate_dataset:
        labels.append(label)
        inputs = inputs.to(device)
        output = model.forward(inputs)
        preds.append(output)
    preds = torch.vstack(preds)
    _, preds = torch.max(preds, 1)
    preds = preds.detach().cpu().numpy()
    labels = torch.hstack(labels).detach().cpu().numpy()
    print(preds.shape)
    print(labels.shape)
    metric = accuracy_score(labels, preds)
    
    return metric


if __name__ == '__main__':
    caltech101, _, num_class = load_caltech101()
    train, test = random_split(caltech101,[0.007, 0.993])
    auroc = train_and_evaluate_model(train, train, 30, 0.9, 0.9, 0.9, 1e-3, 0.99, num_class, 10, torch.device('cpu'))
    print(auroc)