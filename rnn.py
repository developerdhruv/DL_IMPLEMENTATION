import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device is: {device}")
import torch.nn as nn
import torch.optim as optim
class RNN1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super (RNN1 ,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size , embedding_dim) #embedding layer
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first = True) #RNN Layer
        self.fc = nn.Linear(hidden_size, output_size) #Fully connected layer to produce output
    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

#PARAMETERS
vocab_size = 10   #size of vocabulart(max integer index+1)
embedding_dim = 4# Dimensions of the embedding vector
hidden_size = 10#Number of features in hidden state
output_size = 1#number of output class

model = RNN1(vocab_size, embedding_dim, hidden_size, output_size)

#LOSS AND OPTIMIZER
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

#sample data (batch size, sequence length)

inputs = torch.tensor([[1,2,3], [2,3,4]])
targets = torch.tensor([[4.0], [5.0]])

#training loop

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs =model(inputs)
    loss = criterion(outputs , targets)
    loss.backward()
    optimizer.step()

    if (epoch +1 ) %10 == 0:
        print(f'EPOCH [{epoch+1}/100], loss: {loss.item():.4f}')

#testing model
model.eval()
test_input = torch.tensor([[3,4,5]])
predicted = model(test_input) 
print(f'predicted value: {predicted.detach().numpy()}')


















#rnn2
class RNN2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super (RNN2 ,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size , embedding_dim) #embedding layer
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first = True) #RNN Layer
        self.fc = nn.Linear(hidden_size, output_size) #Fully connected layer to produce output
    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

#PARAMETERS
vocab_size = 10   #size of vocabulart(max integer index+1)
embedding_dim = 4# Dimensions of the embedding vector
hidden_size = 10#Number of features in hidden state
output_size = 1#number of output class

model = RNN2(vocab_size, embedding_dim, hidden_size, output_size)

#LOSS AND OPTIMIZER
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

#sample data (batch size, sequence length)

inputs = torch.tensor([[1,2,3], [2,3,4]])
targets = torch.tensor([[[4.0], [5.0], [6.0]], [[5.0],[6.0], [7.0]]])

#training loop

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs =model(inputs)
    loss = criterion(outputs , targets)
    loss.backward()
    optimizer.step()

    if (epoch +1 ) %10 == 0:
        print(f'EPOCH [{epoch+1}/100], loss: {loss.item():.4f}')

#testing model
model.eval()
test_input = torch.tensor([[3,4,5]])
predicted = model(test_input) 
print(f'predicted value: {predicted.detach().numpy()}')








input_size =3
hidden_size = 4
num_layers = 1
batch_size = 1
seq_length = 5

rnn = nn.RNN(input_size,hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

input_tensor = torch.randn(batch_size, seq_length, input_size)

out, hn = rnn(input_tensor)

print(f"shape: ", out.shape)
print(hn.shape)


