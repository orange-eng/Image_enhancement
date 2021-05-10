import torch
pthfile = 'best.pth'
net = torch.load(pthfile)
net["layer1.0.coefficient"] = torch.FloatTensor([0])
net["layer1.1.coefficient"] = torch.FloatTensor([0])

for key, v in enumerate(net):
    print(key, v)

for key, v in net.items():
    if key == 'conv1.weight':
        print(key,v[0][0][0][0])

torch.save(net, "Weights.pth")