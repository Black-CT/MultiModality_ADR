from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("log")
import numpy as np
import time
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
from torch_geometric.loader import DataLoader
import torch
from In_memory_dataset_whole import MyOwnDataset
from model.multi_modality import Net,ablation_study1,ablation_study2,ablation_study3
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

""" parameters """
epochs = 100
train_batch_size = 32
number_of_task=1
""" data set """

our_dataset = MyOwnDataset(root="drug_data/")

train_size = int(0.8 * len(our_dataset))
test_size = len(our_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(our_dataset, [train_size, test_size])

print("size of dataset",len(train_dataset))


""" data loader """
# train_loader = DataLoader(train_dataset, batch_size=train_batch_size,shuffle=True,collate_fn=collate_fn)
# test_loader= DataLoader(test_dataset, batch_size=train_batch_size,shuffle=True,collate_fn=collate_fn)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size,shuffle=True)
test_loader= DataLoader(test_dataset, batch_size=train_batch_size,shuffle=True)
# loader = DataLoader(our_dataset, batch_size=train_batch_size, shuffle=False)
""" load model """
model = Net().to(device)
# model = ablation_study2().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
criterion = torch.nn.MSELoss()

""" train """
train_losses = []
train_acces = []


for epoch in range(epochs):
    start_time = time.time()
    train_loss = 0
    train_acc = 0

    model.train()
    for i, (data) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        smiles=data.smiles
        inputs = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")
        inputs=inputs.to(device)
        outputs = model(data,inputs)
        # print(outputs)
        data.y=data.y.view(-1,number_of_task)
        # print(data.y)
        loss = criterion(outputs, data.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss


    writer.add_scalar('Train/Loss', train_loss / len(train_loader), epoch)

    eval_loss = 0
    eval_acc = 0

    """ evaluate """
    model.eval()

    preds = []
    trues = []
    # 处理方法同上
    # tp = 0
    # tn = 0
    # fp = 0
    # fn = 0

    for i, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            data = data.to(device)
            smiles = data.smiles
            inputs = tokenizer(smiles, padding=True, truncation=True, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(data, inputs)
            data.y = data.y.view(-1,number_of_task)
            loss = criterion(outputs, data.y)
            eval_loss += loss

            labels = data.y.cpu().tolist()
            belief_score = outputs.cpu().tolist()
            preds.extend(belief_score)
            trues.extend(labels)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    print('epoch:{},Train Loss:{:.4f},Test Loss:{:.4f}'
          .format(epoch, train_loss / len(train_loader),
                  rmse))

    stop_time = time.time()

    print("time is:{:.4f}s".format(stop_time-start_time))

