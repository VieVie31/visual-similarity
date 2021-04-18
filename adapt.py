class Adaptation (nn.Module):
    def __init__(self,input):
        super().__init__()
        self.weight_matrix = nn.Linear(input,1024,bias=False)



    def forward(self,x):
        x = self.weight_matrix(x)
        x = torch.relu(x)

        return x

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def calc_loss(left,right,temp):
  sim1 = sim_matrix(left, right)
  sim2 = sim1.t()
  loss_left2right= F.cross_entropy(sim1* temp, torch.arange(len(sim1)).long().to(device)).to(device)
  loss_right2left= F.cross_entropy(sim2* temp, torch.arange(len(sim2)).long().to(device)).to(device)
  loss = loss_left2right* 0.5 + loss_right2left * 0.5
  return loss

def get_topK(left,right,k):
  correct = 0
  sim = sim_matrix(left, right)

  sim2 = sim.t()
  for i in range(len(left)):
    val,j = torch.topk(sim[i],k,sorted=True)
    val2, j2 = torch.topk(sim2[i],k,sorted=True)

    for pos in range(k):
      if j[pos].item()==i | j2[pos].item()==i:

        correct += 1
  return correct

def get_topk2(left,right, k):
  sim = sim_matrix(left, right)

  sorted_idx = sim.argsort(0)


  sens1 = np.array([lbl in sorted_idx[lbl][:k] for lbl in range(len(left))])
  sim = sim.t()
  sorted_idx = sim.argsort(0)
  sens2 = np.array([lbl in sorted_idx[lbl][:k] for lbl in range(len(left))])

  return ((sens1 | sens2).mean())

def train1(left_train, right_train, left_tst, right_tst,epochs,adaptation, optimizer,k,temp):


    score =0
    recall = []
    left_train = torch.from_numpy(left_train).type(torch.FloatTensor).to(device)
    right_train = torch.from_numpy(right_train).type(torch.FloatTensor).to(device)
    left_tst = torch.from_numpy(left_tst).type(torch.FloatTensor).to(device)
    right_tst = torch.from_numpy(right_tst).type(torch.FloatTensor).to(device)

    train_losses =[]
    test_losses=[]
    total_train = len(left_train)
    total_test = len(left_tst)
    for epoch in tqdm(range(epochs)):


        adaptation.train()
        running_loss =0.0
        optimizer.zero_grad()
        left_adapted = adaptation(left_train)
        right_adapted = adaptation(right_train)
        loss = calc_loss(left_adapted,right_adapted,temp)
        loss.backward()
        optimizer.step()
        running_loss +=loss.item()
        with torch.no_grad():
          adaptation.eval()
          test_loss=0.0
          left_adapted_test = adaptation(left_tst)
          right_adapted_test = adaptation(right_tst)
          loss = calc_loss(left_adapted_test,right_adapted_test,temp)
          test_loss += loss.item()

        score = get_topK(left_adapted_test,right_adapted_test,k)
        recall.append(score/total_test)
        train_losses.append(running_loss)
        test_losses.append(test_loss)
        print("Epoch " , (epoch+1))
        print("Train Loss: ",running_loss)
        print("Test Loss: ",test_loss)
        print("Recall :", recall[epoch])

    print(sum(recall)/len(recall))
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Testing loss')
    plt.legend()
    plt.show()
    plt.figure(1)
    plt.plot(recall,label ='Recall' )
    plt.legend()
    plt.show


def run(runs,left,right,epochs, k,temp,type,spl):

  """
  runs : number of experiments
  left, right : embeddings
  epochs : number of epochs
  k : aR@k
  temp : temperature
  type : gradient accumulation =>2
          else => 1
  spl : train size
  """
  for i in range(runs):
    N = len(left)
    sample = int(spl*N)
    idx = np.random.permutation(data['left_ebds'].shape[0])
    train_idx, test_idx = idx[:sample], idx[sample:]

    model = Adaptation(512)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr =0.0005,weight_decay=1e-5)



    left_train, left_test, right_train, right_test = left[train_idx,:], left[test_idx,:], right[train_idx,], right[test_idx,]
    if type==1:
      train1(left_train,right_train,left_test, right_test,epochs, model, optimizer,k,temp)
    else:
      train2(left_train,right_train,left_test, right_test,epochs, model, optimizer,k,temp)
