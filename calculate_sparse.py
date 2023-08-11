import numpy as np
import pandas as pd

# sider27
# data=pd.read_csv("data/sider27.csv")
# matrix=data.iloc[:,1:].values
# # matrix=data.iloc[:,1].values
# smiles=data["smiles"]

#clintox
data=pd.read_csv("data/clintox.csv")
smile=smiles=data["smiles"]
matrix=data.iloc[:,1:].values

#sider5868
# data=pd.read_csv("data/our_sider.csv")
# smile=smiles=data["SMILES"]
# matrix=data.iloc[:,4:].values

# HIV
# data=pd.read_csv("data/HIV.csv")
# smile=smiles=data["smiles"]
# matrix=data.iloc[:,1].values

# BACE
# data=pd.read_csv("data/bace.csv")
# smile=smiles=data["mol"]
# matrix=data.iloc[:,2].values

# BBBP
# data=pd.read_csv("data/BBBP.csv")
# matrix=data.iloc[:,2].values
# smiles=data["smiles"]

total_cells = matrix.size

# count the number of 1s in the DataFrame
num_ones = (matrix == 1).sum().sum()

# calculate the percentage of 1s in the DataFrame
percentage_ones = num_ones / total_cells * 100

print('Percentage of 1s in DataFrame:', percentage_ones, '%')