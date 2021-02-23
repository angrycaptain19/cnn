import torch
import torch.autograd.variable as Variable
import torch.nn as nn
import torch.optim as optim

def make_features(x):
    #建立数据矩阵
    x = x.unsqueeze(1)
    return torch.cat([x**i for i in range(1,4)], 1)

#定义好真实函数，y = 0.9 + 0.5x + 3x^2 + 2.4x^3
w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])

def f(x):
    return x.mm(w_target) + b_target[0]
    #x.mm是矩阵的乘法

"""
print(make_features(torch.tensor([1,2,3])))
>>> tensor([[ 1,  1,  1],
            [ 2,  4,  8],
            [ 3,  9, 27]])
"""

def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)

class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        return self.poly(x)

model = poly_model().cuda() if torch.cuda.is_available() else poly_model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epoch = 0
while True:
    batch_x, batch_y = get_batch()

    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.data[0]

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    epoch += 1
    if print_loss < 1e-3:
        break
