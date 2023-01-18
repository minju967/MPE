import numpy as np
import torch

targets = np.array([[1,0,0,1,0,0],[0,1,0,1,0,0]])
targets = torch.tensor(targets)

outputs = np.array([[0,1,0,1,0,0],[1,0,0,1,0,0]])
outputs = torch.tensor(outputs)

results = torch.eq(targets, outputs)
total = 0
correct = 0
for i in range(results.shape[0]):
    if sum(list(results[i].numpy())) == 6:
        correct += 1
        total += 1
    else:
        total += 1

print(correct/total)



