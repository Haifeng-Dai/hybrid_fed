import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)

print(a.mean())
print(torch.nn.MSELoss()(a, torch.zeros_like(a)))
