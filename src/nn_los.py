import torch
from torch import nn

inputs = torch.tensor([1, 2, 4], dtype=torch.float32)
targets = torch.tensor([1, 2, 7], dtype=torch.float32)




x = torch.tensor([[0.1, 0.2, 0.3]])
y = torch.tensor([1], dtype=torch.long)
loss = nn.CrossEntropyLoss()
resutls = loss(x,y)
print(resutls)