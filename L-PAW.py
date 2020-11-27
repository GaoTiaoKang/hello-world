import torch as t
from torch.autograd import Variable as V
from torch import nn,optim
import torch.nn.functional as F

Lr=0.03
Et=0. 
N=50
epoches=3
T=32
batch_size=128

class SparseAutoencoder(nn.Module):
    output=t.Tensor(50)
    def __init__(self,n_inp,n_hidden):
        super(SparseAutoencoder,self).__init__()
        self.encoder=nn.Linear(n_inp,n_hidden)
        output=self.encoder
        self.decoder=nn.Linear(n_hidden,n_inp)
 
    def forward(self,x):
        encoded=t.sigmoid(self.encoder(x))
        decoded=t.sigmoid(self.decoder(encoded))
        return encoded,decoded

model = SparseAutoencoder(100,50)

if t.cuda.is_available():
      model.cuda()

x = V(t.randn(100))

encode, decode = model(x)

def loss(q):
  sorted, index=t.sort(q,descending=True)
  r=t.split(index, 50, dim=0) 
  for ik in r[0]:
    s1=t.sum(0.2*t.log(0.2/q[ik]))
    s2=t.sum((1-0.2)*t.log((0.8)/(1-q[ik])))
  return s1+s2


optimizier = optim.Adam(model.parameters(), lr=0.03,weight_decay=1e-5)

p=t.Tensor(50)
for epoch in range(epoches):
  for node in range(N):
    #向前传播并计算每个结点的平均激活度
    encode, decode = model(x)
    p[node]=encode[node]/N
  p=V(p,requires_grad=True)
  lossFunction = F.mse_loss(x,decode)+3*loss(p)
  for name,parameter in model.named_parameters():
    if name == 'encoder.weight':
      w=parameter
      w=t.transpose(w, 0, 1)
      #print(w.size())
    if name == 'encoder.bias':
      b=parameter
      #print(b.size())
  CX=model.output
  #x_output=t.mm(x,w)
  #print(x_output)
  # 更新网络中的参数
  optimizier.zero_grad()
  lossFunction.backward(retain_graph=True)
  optimizier.step()

loss = nn.MSELoss()

data_iter = t.utils.data.DataLoader(
       t.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)
for i in range(epoches):
  #分段设置衰退率
  if i<10:
    Et=1
  else if i<20:
    Et=0.8
  else if i<40:
    Et=0.6
  else if i<60:
    Et=0.4
  else if i<80:
    Et=0.2
  else
    Et=0.01
  #训练RNN-GRU神经网络
  for t in T
    rnn = nn.GRU(128, 128, 2)
    #input = V(t.randn(5, 3, 10))
    #h0 = V(t.Tensor(128))
    #output, hn = rnn(CX, h0)
  #应用小批量随机梯度下降法进行后向传播
    for batch_size, (x, y) in enumerate(data_iter):
      l = loss(rnn(X).view(-1), y) 
      optimizer.zero_grad()
      l.backward()
      optimizer.step()
 
  
    
