import torch
import torchvision
import torchvision.transforms as transforms

from model import ClassifierModel

import numpy as np

# Load CIFAR 10 dataset

# Normalize images to [-1, 1] (torchvision dataset is in range [0, 1])
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Batch size defines how many images are in one datapoint
b_size = 4

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=b_size,
                                         shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Load trained model
net = ClassifierModel()
net.load_state_dict(torch.load('./cifar_classifier.pth'))

# Iterate over dataset and calculate the accuracy of each class separately
predicted_class = [0] * 10
correct_class = [0] * 10
with torch.no_grad():
    for _, data in enumerate(testloader):
        # Load images and labels (batch_size at once)
        images, labels = data

        # todo: Evaluate model with current input
        outputs = net(images);

        # todo: Choose the predicted class as the one with the maximum response
        pred_class = torch.argmax(outputs,1);
        # todo: Decide if the network was correct
        clas = (pred_class == labels).squeeze();
        # todo: For each image in mini batch, update corresponding class accuracies
        for i in range(b_size):
            l = labels[i]
            predicted_class[l] += clas[i].item();
            correct_class[l] += 1;

# todo: Print all class accuracies
for i in range(len(classes)):
    # ...
    print(classes[i] + "  " + str(predicted_class[i]/correct_class[i]));