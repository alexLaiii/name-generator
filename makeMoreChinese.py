#function that build the whole dataset
def build_dataset(words):
        #Look up table example
        #Use 3 character to predict next character
        X,Y = [],[]
        for w in words:
            context = [end_t] * block_size
            for ch in w + '.':
                ix = stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
           
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        
        #X.shape = [num of examples,3] Y.shape = [num of examples], dtype: torch.int64
        return X, Y


import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random


         
if __name__ == "__main__":
    
    words = open("ChineseName.txt", "r", encoding="utf-8").read().splitlines()
    #get all the chars but chinese
    chars = sorted(list(set(''.join(words))))
     #mapping all the charcter to integer

    
    stoi = {val: idx for idx, val in enumerate(chars)}
    stoi["."] = len(stoi)
    itos = {idx : val for val, idx in stoi.items()}
    
    #context length: how many characters do we take to predict the next one?
    end_t = len(stoi) - 1
    block_size = 2

    random.seed(323123)
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    #split the dataset
    #597740 example in the training set
    X_tr, Y_tr = build_dataset(words[:n1])
    X_val, Y_val = build_dataset(words[n1:n2])
    X_test, Y_test = build_dataset(words[n2:])

    
    # Generator for reproducing purpose, make it test and comparable
    g = torch.Generator().manual_seed(21474869)
    
    # decide the dimension -> if one hot = 27
    possible_output = len(itos)
    dimension= 20
    m_neurons = 200
    
   
    # weight embedding matrix, input layer    
    C = torch.randn([possible_output, dimension], generator = g, requires_grad = True)

    # hidden layer 
    W1 = torch.randn([int(dimension * block_size), m_neurons], generator = g, requires_grad = True)
    # bias
    b1 = torch.randn([m_neurons], generator = g, requires_grad=True)

    # output layer
    # W2 = torch.randn([m_neurons,possible_output], generator = g, requires_grad=True)
    # b2 = torch.randn([possible_output], generator=g, requires_grad=True)

    W2 = torch.randn([m_neurons, possible_output], generator = g, requires_grad=True)
    b2 = torch.randn([possible_output], generator=g, requires_grad=True)

 




    # Train dataset
    for i in range(300000):
        # mini-batch to optimize training efficiency
        ix = torch.randint(0, X_tr.shape[0], (32,))

        # forward pass
        emb = C[X_tr[ix]]  # shape = [batch_size, number of inputs, dimensions]
        # emb.shape = [32,3,12]
        # pass to hidden Layer
        h = F.tanh(emb.view([emb.shape[0], int(block_size * dimension)]) @ W1 + b1)
  
        logits = h @ W2 + b2

        loss = F.cross_entropy(logits, Y_tr[ix])


        # set learning rate
        # lr = lr_exp[i]
        lr = 0.1
        if(i > 90000):
             lr = 0.01
        if(i > 150000):
             lr = 0.001
        if(i > 200000): 
             lr = 0.0001
        # set gradient back to zero
        param = [C,W1,b1,W2,b2]
        for p in param:
             p.grad = None

        # backward pass
        loss.backward()

        # Adjustment
        for p in param:
             p.data += -lr * p.grad




    # Cal training loss
    emb = C[X_tr]
    h = torch.tanh(emb.view(X_tr.shape[0], int(dimension * block_size)) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y_tr)
    print("Training loss: ", loss.item())

    # Cal validation loss
    emb = C[X_val]
    h = torch.tanh(emb.view(X_val.shape[0], int(dimension * block_size)) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y_val)
    print("Validation loss: ", loss.item())



    # Sampling
    g2 = torch.Generator().manual_seed(1242112)
    for i in range(30):
         out = []
         context = [end_t] * block_size
         while True:
              emb = C[torch.tensor([context])]
              h = torch.tanh(emb.view(1,-1) @ W1 + b1)
              logits = h @ W2 + b2
              probs = F.softmax(logits, dim = 1)
              ix = torch.multinomial(probs, num_samples=1, generator= g2).item()
              context = context[1:] + [ix]
              out.append(itos[ix])
              if ix == end_t:
                break
         print("".join(out))    