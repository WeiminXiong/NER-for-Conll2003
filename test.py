import torch

a = torch.nn.Parameter(torch.full((3, 3), 3 , dtype=torch.float))
b = a[:,2]
c = torch.full((3, ), 4)
print(a[-1:])