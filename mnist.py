import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import time


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1)
        self.fc1   = nn.Linear( 6144,4096)
        self.fc2   = nn.Linear( 4096,4096) 
        self.fc3   = nn.Linear( 4096,10)
        self.dropout = nn.Dropout(0.25)
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # forward
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), 6144 ) #before fc, equal np.resize(), where size(0) means batchsize
        nn.Dropout()
        x = F.relu(self.fc1(x))
        nn.Dropout()
        x = F.relu(self.fc2(x))
        nn.Dropout()
        x = F.relu(self.fc3(x))
        #x = self.softmax(x)
        x = F.log_softmax(x, dim=1)
        return x

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

model = myModel()
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 10
train_losses, test_losses = [], []

#train
print("use_cuda: ", use_cuda)

for e in range(epochs):
    epoch_start = time.time()
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        #images = images.cuda()
        images = images.to(device)
        labels = labels.to(device)
        
        # TODO: Training pass
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0
        
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
                
            log_ps = model(images)
            test_loss += criterion(log_ps, labels)
                
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
                
    train_losses.append(running_loss/len(trainloader))
    test_losses.append(test_loss/len(testloader))

    epoch_end = time.time()
    torch.save(model.state_dict(), 'logs/'+ str(e+1) +'.pth')

    print("Epoch: {}/{} ".format( e+1, epochs),
          "Consuming: {:.2f}s ".format(epoch_end - epoch_start),
          "Training Loss: {:.3f} ".format(running_loss/len(trainloader)),
          "Test Loss: {:.3f} ".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
      
        
import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)

