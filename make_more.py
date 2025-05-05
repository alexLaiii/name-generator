import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

if __name__ == "__main__":
    words = open("names.txt", "r").read().splitlines()
    # get all the chars and sort it, etc: a,b,c,d,e,...
    chars = sorted(list(set(''.join(words))))
    #  mapping all the charcter to integer
    stoi = {val: idx+1 for idx, val in enumerate(chars)}
    stoi['.'] = 0
    itos = {idx : val for val, idx in stoi.items()}




    """
    split dataset into 3 training set to prevent overfitting
    val_loss determine is the one who determine how good your model is

    Roughly split as
    |80%|10%|10%|
    Concept:
    -Dev set(aka validation set) is used during taining to compute val_loss -> it is never used for weight updates, only for evalutaion
    - val_loss helps develop overfitting
        - If training loss decrease, but val_loss increase or flat -> model is starting to memorize (bad)
        - A rising or stagnant val_loss means generalization is getting worse
    - Early stopping = stop training when val_loss stops improving
        - Common rule: stop after patience steps (e.g 3-5) with no improvement
        - Prevents wasting time and avoids overfitting
    - Training loop logic
        - Track the best val_loss so far
        -If val_loss improves -> reset patience counter
        -If it doesn't -> increment patience counter
        -Stop training when counter hits the patience threshold
    - This method ensures the model learns just enough from training data without memorizing it
    
    """
    block_size = 3 # context length: how many characters do we take to predict the next one?
    #function that build the whole dataset
    def build_dataset(words):
        # Look up table example
        # Use 3 character to predict next character
        X,Y = [],[]
        for w in words:
            #print(w)
            context = [0] * block_size
            for ch in w + '.':
                ix = stoi[ch]
                X.append(context)
                Y.append(ix)
                #print(''.join(itos[i] for i in context), '---->', itos[ix])
                context = context[1:] + [ix]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        # X.shape = [32,3] Y.shape = [32], dtype: torch.int64
        return X, Y
    
    random.seed(23)
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))
    print(n1,n2,len(words))
            
    # split the dataset
    X_tr, Y_tr = build_dataset(words[:n1])
    X_val, Y_val = build_dataset(words[n1:n2])
    X_test, Y_test = build_dataset(words[n2:])

    print(len(itos))
    print(stoi)
    print(itos)



        
    # Generator for reproducing purpose, make it test and comparable
    g = torch.Generator().manual_seed(21474869) 
    #look-up table C embedding
    # aka input layer construct

    dimension = 12
    C = torch.randn([27,dimension], generator=g, requires_grad=True) # each one of 27 characters will have a 2 dimensional embeddings
    # all_row = C[torch.arange(27)] # it can retrieve mutiple row from C using list like C[[5,6,7,8]], will retrieve row 5,6,7,8 at once
    # Hidden layer, 6 is the input from the first layer, 3 characters with 2 dimesnion each
    # when definding a layer, the second dimension is always the number of output you want, if its not force defined

    # second layer construct
    W1 = torch.randn([int(dimension * block_size),300], generator=g, requires_grad = True)
    # The bias
    b1 = torch.randn([300], generator=g, requires_grad = True)

    # 3rd layer construct
    W2 = torch.randn([300, 27], generator=g, requires_grad = True)
    b2 = torch.randn([27], generator=g, requires_grad = True)

    # show the number of parameters use
    print(C.nelement() + W1.nelement() + b1.nelement() + W2.nelement() + b2.nelement() )

    # learning rate test
    lrs = torch.linspace(-3, -1, 250000) #this return 1000 numbers between 0.001 and 1
    lr_exp = 10 ** lrs

    
    lr_tracker, loss_tracker = [], []

    stepi, lossi = [], []



    for i in range(300000):

        # minibatch construct
        ix = torch.randint(0,X_tr.shape[0],(32,)) #X.shape[0] indicates how many example for efficiency


        # forward pass
        emb = C[X_tr[ix]] # C[X][ix] is a [32,3,12]
        

        # Pass to the hidden layer
        # emb.view force the matrix to view as the given dimension, emb.shape[0] return 32 because emb is  [32,3,2], the first dimension is 32
        # Here mb.view([emb.shape[0],6]) @ W1 = [32,100], b1 = [100]
        # 32, 100
        #  1, 100 -> broadcoasting is well defined
        h = F.tanh(emb.view([emb.shape[0],int(dimension * block_size)]) @ W1 + b1) # here is same as emb.view([32,6]) @ W1 + b1

        # h@W2 is well defined since
        # 32 ,100
        # 100,27
        logits = h @ W2 + b2 # logits now is a [32,27]
        # F.cross_entropy to do softmax and probs calculation all together
        loss = F.cross_entropy(logits, Y_tr[ix])


        # backward pass
        C.grad = None
        W1.grad = None
        W2.grad = None
        b1.grad = None
        b2.grad = None
        loss.backward()
        # lr = -lr_exp[i]
        lr = -0.1
        if(i > 90000):
            lr = - 0.01
        if(i > 150000):
            lr = -0.001
        if(i > 200000 ):
            lr = -0.0001
      
        C.data += lr  * C.grad
        W1.data += lr * W1.grad
        W2.data += lr * W2.grad
        b1.data += lr * b1.grad
        b2.data += lr * b2.grad
        # lr_tracker.append(lrs[i])
        # loss_tracker.append(loss.item())
        stepi.append(i)
        lossi.append(loss.item())


        
    # The total loss
    emb = C[X_tr]
    h = torch.tanh(emb.view(X_tr.shape[0], int(dimension * block_size)) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y_tr)
    print("Train loss: ", loss.item())

    # The value loss 
    emb = C[X_val]
    h = torch.tanh(emb.view(X_val.shape[0], int(dimension * block_size)) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y_val)
    print("Val loss: ", loss.item())

    # Difference will never become zero
    # Because since we have only 5 words as training set
    """
    for example: the first index indicies "..." -> "e" is suppose to be predicted, but in the second index "..." -> "o" is expected
    theres no one case that actually outweights than the other one, therefore, the prediction differences in the first index always exist, 
    and "..." -> "a" would never be guranteen predicted no matter how we trained it with this tiny dataset
    """

    """
    Test which learning rate is suitable
    """
    # see the result
    # plt.plot(lr_tracker, loss_tracker)
    # plt.show()


    ### show how character group (only works when dimension = 2)
    # plt.figure(figsize=(8,8))
    # plt.scatter(C[:,0].data, C[:,1].data, s= 200)
    # for i in range(C.shape[0]):
    #     plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha = "center", va = "center", color="white")
    # plt.grid('minor')
    # plt.show()

    #Sampling
    g2= torch.Generator().manual_seed(3214124)
    for i in range (20):
        out =[]
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim = 1)
            ix = torch.multinomial(probs, num_samples=1, generator= g2).item()
            context = context[1:] + [ix]
            out.append(itos[ix])
            if ix == 0:
                break
        print("".join(out))
            
