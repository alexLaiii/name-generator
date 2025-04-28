import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

#read the feeding dataset
words = open("names.txt", "r").read().splitlines()
#get all the chars and sort it, etc: a,b,c,d,e,...
chars = sorted(list(set(''.join(words))))
#map all the characters to integer
stoi = {val: idx+1 for idx, val in enumerate(chars)}
# '.' indicates the first and last character
stoi['.'] = 0
itos = {idx: val for val, idx in stoi.items()}

# setup W = the weight matrix

g = torch.Generator().manual_seed(10000)
W = torch.randn((27,27), generator= g, requires_grad=True)

xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.'] # "." represents start and end of the word
    for ch1, ch2 in zip(chs, chs[1:]):
        row = stoi[ch1]
        col = stoi[ch2]
        xs.append(row)
        ys.append(col)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement() # store how many example in hs total
# xs and ys must have the same number of elements

x_enc = F.one_hot(xs, num_classes=27).float()
# forward pass
for i in range(100):

    logits = x_enc @ W # retreive the correct row for W, example x_enc is in 'e', then retrieve the row that represent "e*"
    counts = logits.exp()
    probs = counts/counts.sum(1, keepdim=True) # 1 means sum accross dimension 1(the columns)
    loss = -probs[torch.arange(num),ys].log().mean() + 0.01*(W**2).mean()
    #backward pass
    W.grad = None
    loss.backward() # calculate the gradient for each entries in W
    # adjustment
    W.data += -50 * W.grad
    print(f"{i}: {loss}")
 
# Sampling from the neural net
if __name__ == "__main__":
    for i in range(5):
        out= []
        ix = 0
        while True:
            # xenc @ W will retreive the ix row of W, represent the probabilites
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float() # torch.tensor([ix]) -> tensor([ix])
            logits = xenc @ W # retreive the correct row for W, example x_enc is in 'e', then retrieve the row that represent "e*"
            # softmax here to get the probability
            counts = logits.exp()
            probs = counts/counts.sum(1, keepdim=True) # 1 means sum accross dimension 1 (the columns)
            ##
            ix = torch.multinomial(probs, num_samples = 1, replacement=True, generator = g).item()
            out.append(itos[ix])
            if ix == 0: # encounter ".", end of word
                break
        
        print(''.join(out)) # ''.join() joins a list of characters or list of strings into a single string