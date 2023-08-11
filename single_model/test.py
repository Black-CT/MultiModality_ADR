import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool,GATConv,SAGEConv,Sequential,GraphNorm,global_max_pool,DeepGCNLayer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoModelForSequenceClassification
import torch
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # after tokenizer, edit this link
        self.model = AutoModelForSequenceClassification.from_pretrained("gumgo91/IUPAC_BERT")
        # try IUPAC without pretraining

        # load model from huggingface
        # self.model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")

        # # load from local cache
        # model_path = os.path.join(os.getcwd(), "model/local_cache/distilroberta")
        # self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        # # self.model = AutoModelForMaskedLM

        self.outputlayer = nn.Sequential(
            nn.Linear(50265, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
            # nn.Sigmoid(),
        )
        self.decodelayer1=nn.TransformerDecoderLayer(d_model=50265, nhead=15,batch_first=True)
        # self.decodelayer2 = nn.TransformerDecoderLayer(d_model=767, nhead=13, batch_first=True)




    def forward(self,inputs):
        # memory=self.gnn_branch(data)
        # memory=memory.view(-1,4,767)
        tgt_mask=inputs.attention_mask.to(torch.float)
        tgt_key_padding_mask=tgt_mask
        tgt_key_padding_mask=torch.where(tgt_key_padding_mask==0,1,0)
        outputs = self.model(**inputs)
        tgt=outputs[0]
        final_representation=self.decodelayer1(tgt,memory,tgt_key_padding_mask=tgt_key_padding_mask)
        # final_representation = self.decodelayer2(final_representation, memory, tgt_key_padding_mask=tgt_key_padding_mask)
        final_representationfinal_representation=final_representation[:,0,:]

        x=self.outputlayer(final_representationfinal_representation)
        return x


if __name__ == '__main__':
    IUPAC = ["N-(4-hydroxyphenyl)ethanmide",
             "(2S)-1-[(2S)-2-[[(2S)-1-ethoxy-1-oxo-4-phenylbutan-2-yl]amino]-6-[(2,2,2-trifluoroacetyl)amino]hexanoyl]pyrrolidine-2-carboxylic acid",
             "1-methyl-1-nitroso-3-[(3R,4R,5S,6R)-2,4,5-trihydroxy-6-(hydroxymethyl)oxan-3-yl]urea"]

    model = AutoModelForSequenceClassification.from_pretrained("gumgo91/IUPAC_BERT")
    tokenizer = AutoTokenizer.from_pretrained("gumgo91/IUPAC_BERT")


    inputs = tokenizer(IUPAC, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(inputs)

    # x=torch.randn(3,9)
    # edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long)
    # # drug = ["C[N+](C)(C)CC(CC(=O)O)O"]
    # drug = ["C[N+](C)(C)CC(CC(=O)O)O", "C[N+](C)(C)CC(CC(=O)O)O", "C[N+](C)(C)CC(CC(=O)O)OCCCCCCC"]
    # tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    # inputs = tokenizer(drug, padding=True, truncation=True, return_tensors="pt")
    # # inputs = {k: v.to(device) for k, v in inputs.items()}
    # print(inputs)
    #
    # data1=Data(x=x,edge_index=edge_index,id=1)
    # data2 = Data(x=x, edge_index=edge_index, id=1)
    # data3 = Data(x=x, edge_index=edge_index, id=1)
    # data_list=[data1,data2,data3]
    # loader=DataLoader(data_list,batch_size=3)
    # for i,mini_batch in enumerate(loader):
    #     print(mini_batch)
    # # print(data)
    # # gnn_branch=GNN_branch()
    # # gnn_out=gnn_branch(data)
    # # print(gnn_out.shape)
    # # print(torch.unsqueeze(gnn_out,2).size())
    #     net=ablation_study3()
    #     out=net(mini_batch,inputs)
    #     print(out.size())