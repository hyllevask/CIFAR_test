import torch, torchvision


train_data = torchvision.datasets.CIFAR10(
    "datasets",
    download = True,
    transform = torchvision.transforms.ToTensor()
    )

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size = 32,
    shuffle = True,
    num_workers = 32
    )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 50

network = torchvision.models.resnet18(pretrained=True)
torch.nn.Linear(in_features=512, out_features=10, bias=True)
network._modules['fc'] = torch.nn.Linear(in_features=512, out_features=10, bias=True)
network.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

import tqdm

p_bar = tqdm.tqdm(total = len(train_loader), desc='Batch')

print(device)

for epoch in range(epochs):
    p_bar = tqdm.tqdm(total = len(train_loader), desc='Batch')
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        p_bar.update(1)
        img,label = data[0].to(device),data[1].to(device)
        optimizer.zero_grad()


        out = network(img)
        loss = criterion(out,label)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()


torch.save(network.state_dict(),'saved_model.pth')
