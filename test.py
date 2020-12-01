#test.py

import torch, torchvision,numpy


test_data = torchvision.datasets.CIFAR10(
    "datasets",
    download = True,
    transform = torchvision.transforms.ToTensor(),
    train=False
    )

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size = 4,
    shuffle = True,
    num_workers = 4
    )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = torchvision.models.resnet18()
network.load_state_dict(torch.load('saved_model.pth',map_location=torch.device('cpu')))
network.eval()
criterion = torch.nn.CrossEntropyLoss()

for i,data in enumerate(test_loader):
    img,label = data
    out = network(img)
    print(label)
    print(out.detach().numpy().argmax())
    break