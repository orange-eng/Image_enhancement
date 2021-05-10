from torchsummary import summary
from torchvision.models import resnet18
from torchvision.models import mobilenet_v2,inception_v3
import torch
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)
#model = inception_v3()
#summary(model, input_size=[(3, 256, 256)], batch_size=2, device=device)

network = mobilenet_v2()
i = 0
for name, module in network.named_modules():
    i = i +1
    if i == 1:
        print(i,module)
    # for p_name, p_tensor in module.named_parameters():
    #     print(p_name)
