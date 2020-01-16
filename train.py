import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import model as m

# Load CIFAR 10 dataset

# Normalize images to [-1, 1] (torchvision dataset is in range [0, 1])
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Batch size defines how many images are in one datapoint
b_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size,
                                          shuffle=True, num_workers=0)

# todo: initialize network and copy to cuda

net = m.ClassifierModel()
# todo: Define cross entropy loss and SGD optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

device = torch.device("cuda")
# Train for 2 epochs
for epoch in range(2):

    # Init loss
    train_loss = 0.0

    # Iterate over dataset and get batches
    for i, data in enumerate(trainloader):
        inputs, labels = data

        # todo: Copy data to cuda
        inputs, labels = inputs.to(device), labels.to(device)

        # todo: Zero the gradients
        optimizer.zero_grad()

        # todo: Forward: Evaluate model with current input
        net.cuda()
        output = net(inputs)

        # todo: Backward: Calculate loss and gradients
        loss = criterion(output, labels)
        loss.backward()

        # todo: Optimize: Update weights of network
        optimizer.step()

        # Print statistics
        train_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, train_loss / 2000))
            train_loss = 0.0

print('Finished Training')

# Save model
torch.save(net.state_dict(), './cifar_classifier.pth')

print('Model saved')
