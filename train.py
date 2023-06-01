import json 
from nltk_utils import * 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mymodel import *

with open('intents.json','r') as file_object:
    intents = json.load(file_object)

total_words = []
tags = []
xy_data = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        total_words.extend(w)
        xy_data.append((w,tag))
    
ignore_words = ['?','!','.',',']
total_words = [steming(word) for word in total_words if word not in ignore_words]
total_words = sorted(set(total_words))
tags = sorted(set(tags))
#print(total_words)
#print(tags)

X_train = []
y_train = []
for (pattern_sentence, tag) in xy_data:
    bag = bag_of_words(pattern_sentence, total_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train 
        self.y_data = y_train
        
    #dataset with index
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.n_samples 

batch_size = 8    
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000
dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#LossOptimizor 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        labels = labels.to(dtype=torch.long)
        output = model(words)
        loss = criterion(output, labels)
        
        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1)% 100 == 0:
            print(f"epoch {epoch+1}/{num_epochs}, loss={loss.item():.5f}")

print(f"Final Loss{loss.item():.5f}")

data = {
    "model_state":model.state_dict(),
    "input_size": input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "total_words":total_words,
    "tags":tags
}
FILE = "data.pth"
torch.save(data, FILE)
print(f"Training Completed. File saved to {FILE}")
