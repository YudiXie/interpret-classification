# %%
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# %%
def pdf_gauss(x, mu, sigma):
    return np.exp(- (x - mu)**2 / (2.0 * sigma**2)) / np.sqrt(2.0 * np.pi * sigma**2)

def posterior_c1(x, mu1, mu2, sigma1, sigma2):
    return 1 / (1 + (sigma1 / sigma2) * np.exp((sigma2**2 * (x - mu1)**2 - sigma1**2 * (x - mu2)**2) / (2 * sigma1**2 * sigma2**2)))

def posterior_c2(x, mu1, mu2, sigma1, sigma2):
    return 1 / (1 + (sigma2 / sigma1) * np.exp((sigma1**2 * (x - mu2)**2 - sigma2**2 * (x - mu1)**2) / (2 * sigma1**2 * sigma2**2)))

# %%
class GaussianMixtureDataset(Dataset):
    def __init__(self, m1, s1, m2, s2, num_samples=100000):
        self.num_samples = num_samples
        self.class1_mean = m1
        self.class1_std = s1
        self.class2_mean = m2
        self.class2_std = s2

        # Generate all data at initialization
        self.data = []
        self.labels = []
        for _ in range(num_samples):
            class_label = np.random.randint(0, 2)
            if class_label == 0:
                sample = np.random.normal(loc=self.class1_mean, scale=self.class1_std)
            else:
                sample = np.random.normal(loc=self.class2_mean, scale=self.class2_std)
            self.data.append(sample)
            self.labels.append(class_label)
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Define the neural network
class MyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
# example 1
m1 = 0
s1 = 1

m2 = 3
s2 = 1

# %%
# example 2
# m1 = 0
# s1 = 2

# m2 = 1
# s2 = 1

# %%
# example 3
# m1 = 0
# s1 = 2

# m2 = 3
# s2 = 1

# %%
x_start = -4
x_end = 7

x = np.linspace(x_start, x_end, 100)

y1 = pdf_gauss(x, m1, s1)
y2 = pdf_gauss(x, m2, s2)

fig, ax= plt.subplots()
ax.plot(x, y1, label='Class 1 pdf')
ax.fill_between(x, y1, alpha=0.3)
ax.plot(x, y2, label='Class 2 pdf')
ax.fill_between(x, y2, alpha=0.3)
ax.plot(x, (y1 + y2), label='1 + 2 Mixture')
ax.legend(loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('X Input')
fig.savefig(f'./figures/pdf_m1_{m1}_s1_{s1}_m2_{m2}_s2_{s2}.pdf', transparent=True, bbox_inches='tight')


# %%
def collect_model_predictions(model, input_x):
    # collect model predictions
    inputs = torch.tensor(input_x).float().unsqueeze(1)

    model.eval()
    with torch.inference_mode():
        outputs = model(inputs)
    model.train()

    # Convert outputs to probabilities
    probabilities = torch.softmax(outputs, dim=1)
    prob_class1 = probabilities[:, 0].numpy()
    prob_class2 = probabilities[:, 1].numpy()
    return prob_class1, prob_class2


dataset = GaussianMixtureDataset(m1, s1, m2, s2, 100000)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

test_dataset = GaussianMixtureDataset(m1, s1, m2, s2, 10000)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# Initialize the model, loss function, and optimizer
input_size = 1
hidden_size = 64
output_size = 2
model = MyNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

class_prob = {}

# Training loop
max_batches = 50000
batch_num = 0
running_loss = 10.0
while batch_num < max_batches:
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs.float().unsqueeze(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_num += 1
        running_loss = loss.item() * 0.1 + running_loss * 0.9
        
        if batch_num in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
            # test the model
            model.eval()
            correct = 0
            total = 0
            with torch.inference_mode():
                for inputs, labels in test_dataloader:
                    outputs = model(inputs.float().unsqueeze(1))
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            model.train()
            print(f'Batch {batch_num}, Loss: {running_loss:.4f}, Test Accuracy: {100 * correct / total:.2f}%')

            # collect model predictions
            class_prob[batch_num] = collect_model_predictions(model, x)

        if batch_num >= max_batches:
            break
print('Finished Training')

# %%
# Plot model and theoretical posteriors
def plot_model_vs_theoretical(ax, x, prob_class1, prob_class2):
    ax.plot(x, prob_class1, label='Class 1: Model', linestyle='--', color='C0')
    ax.plot(x, posterior_c1(x, m1, m2, s1, s2), label='Class 1: Theoretical', alpha=0.7, color='C0')
    ax.plot(x, prob_class2, label='Class 2: Model', linestyle='--', color='C1')
    ax.plot(x, posterior_c2(x, m1, m2, s1, s2), label='Class 2: Theoretical', alpha=0.7, color='C1')

    ax.set_xlabel('X Input')
    ax.set_ylabel('Probability')
    ax.legend(fontsize='small', loc='center left')

for batch_num, (prob_class1, prob_class2) in class_prob.items():
    fig, ax = plt.subplots()
    plot_model_vs_theoretical(ax, x, prob_class1, prob_class2)
    ax.set_title(f'Theoretical Posterior vs Model Output at Batch {batch_num}')
    fig.savefig(f'./figures/posterior_m1_{m1}_s1_{s1}_m2_{m2}_s2_{s2}_batch_{batch_num}.pdf', transparent=True, bbox_inches='tight')

# %%



