from model import Model
from dataset import *
from torchvision import datasets
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def train_and_evaluate_model(
    train_dataset,
    validate_dataset,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(graident, square), eps=eps, weight_decay=weight_decay)
    cross_entropy = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        model.train()
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        running_loss = 0.0
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
            print(f'batch {batch}')
            batch += 1
        epoch_loss = running_loss / len(train_data_loader)

        print('train - epoch: {} loss: {:.4f}'.format(epoch,
                                                        epoch_loss))

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
    preds = torch.vstack(preds).detach().numpy()
    labels = torch.hstack(labels).detach().numpy()
    print(preds.shape)
    print(labels.shape)
    auroc = accuracy_score(labels, np.argmax(preds, axis=1))
    
    return auroc


if __name__ == '__main__':
    caltech101, _, num_class = load_caltech101()
    train, test = random_split(caltech101,[0.007, 0.993])
    auroc = train_and_evaluate_model(train, train, 30, 0.9, 0.9, 0.9, 1e-3, 0.99, num_class, 10, torch.device('cpu'))
    print(auroc)