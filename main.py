import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


model = SimpleNN(input_dim=1, output_dim=1).cuda()

alpha = 2; beta = 3
x = np.linspace(0, 4, 100)
y = alpha * x + beta + np.random.randn(100) * 0.3
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

inputs = Variable(torch.from_numpy(x.astype('float32')).cuda())
labels = Variable(torch.from_numpy(y.astype('float32')).cuda())

print(list(model.named_parameters()))


for epoch in range(500):
    z = model(inputs).data.cpu().numpy()
    plt.plot(x, model(inputs).data.cpu().numpy(), color="grey")
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

print(list(model.named_parameters()))


plt.plot(x, y, color="r")
plt.plot(x, outputs.data.cpu().numpy(), color="b")
plt.show()
