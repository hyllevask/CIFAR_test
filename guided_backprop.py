#guided_backprop
# Inspired by https://github.com/utkuozbulak/pytorch-cnn-visualizations/
import torch, torchvision,numpy
import matplotlib.pyplot as plt
from torch.nn import ReLU

test_data = torchvision.datasets.CIFAR10(
    "datasets",
    download = True,
    transform = torchvision.transforms.ToTensor(),
    train=False
    )

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size = 1,
    shuffle = True,
    num_workers = 4
    )

class GetGradient():
    def __init__(self,network):
        self.net = network
        self.net.eval()
        self.gradients = None
        self.update_relus()
        self.hook_layers()
        self.forward_relu_outputs = []

    def hook_layers(self):
        def hook_function(module,grad_in,grad_out):
            self.gradients = grad_in
            print("hej")
            #print(grad_in)
            #print(grad_out)

        print("Registering Hook")
        first_layer = list(self.net._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        def relu_backward_hook_function(module,grad_in,grad_out):
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0],min=0.0)
            del self.forward_relu_outputs[-1]
            return (modified_grad_out,)

        def relu_forward_hook_function(module,tensor_in, tensor_out):
            self.forward_relu_outputs.append(tensor_out)

        for pos, module in self.net._modules.items():
            print(pos)
            if isinstance(module, ReLU):
                print("ReLu")
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)
            #elif 

    def calculate_gradients(self,img,label):
        out = self.net(img)
        pred = out.detach().numpy().argmax(axis=1)
        print(label)
        print(pred)

        network.zero_grad()
        out[0][label].backward()
        return self.gradients[1].data.numpy()[0]
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = torchvision.models.resnet18()
network._modules['fc'] = torch.nn.Linear(in_features=512, out_features=10, bias=True)
network.load_state_dict(torch.load('saved_model.pth',map_location=torch.device('cpu')))


GG = GetGradient(network)


#print(network)

for i,data in enumerate(test_loader):
    img,label = data
    out = network(img)
    gradients = GG.calculate_gradients(img,label)

    s_img = numpy.array(numpy.squeeze(img))
    plt.imshow(numpy.moveaxis(s_img,0, -1))
    plt.savefig('img%i' %i)
    plt.imshow(gradients[0] + gradients[1] + gradients[2])
    plt.savefig('grad%i' %i)
    if i == 00:
        break