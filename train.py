import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models
import matplotlib.pyplot as plt
import time
import os

from data import data_transforms

plt.ion()

data_dir = ''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
check_interval = 5
batchsize = 20
checkpoints = 'checkpoints'
num_epochs = 100

if not os.path.exists(checkpoints):
    os.makedirs(checkpoints)

trainset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                data_transforms['train'])

testset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(testset, batch_size=15,
                                         shuffle=False, num_workers=4)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


best_acc = 0.0
since = time.time()

for epoch in range(100):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    running_loss = 0.0
    running_corrects = 0

    model_ft.train()
    for inputs, labels in trainloader:
        # Iterate over data.
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer_ft.zero_grad()

        # forward

        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        # backward

        loss.backward()
        optimizer_ft.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(trainset)
    epoch_acc = running_corrects.double() / len(trainset)
    print('{} {} Loss: {:.4f} Acc: {:.4f}'.format(
        epoch, 'train', epoch_loss, epoch_acc))

    model_ft.eval()

    test_loss = 0.0
    test_corrects = 0
    for inputs, labels in testloader:
        # Iterate over data.
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # statistics
        test_loss += loss.item() * inputs.size(0)
        test_corrects += torch.sum(preds == labels.data)

    test_loss = test_loss / len(testset)
    test_acc = test_corrects.double() / len(testset)
    print('{} {} Loss: {:.4f} Acc: {:.4f}'.format(
        epoch, 'test', test_loss, test_acc))

    if test_acc > best_acc:
        best_acc = test_acc.item()
    print('best_acc=', best_acc)
    if epoch % check_interval == 0:
        torch.save(model_ft.state_dict(), os.path.join(checkpoints, str(epoch) + '_params.pth'))
        print('model saved')
    print('-' * 10)

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))
