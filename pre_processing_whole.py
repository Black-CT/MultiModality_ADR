import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pubchempy

# sider27
# data=pd.read_csv("data/sider27.csv")
# matrix=data.iloc[:,1:].values
# # matrix=data.iloc[:,1].values
# smiles=data["smiles"]

#clintox
# data=pd.read_csv("data/clintox.csv")
# smile=smiles=data["smiles"]
# matrix=data.iloc[:,1:].values

#sider5868
data=pd.read_csv("data/our_sider.csv")
smile=smiles=data["SMILES"]
matrix=data.iloc[:,4:].values

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

# Lipophilicity
# data=pd.read_csv("data/Lipophilicity.csv")
# matrix=data.iloc[:,1].values
# smiles=data["smiles"]

# ESOL
# data=pd.read_csv("data/delaney-processed.csv")
# matrix=data.iloc[:,1].values
# smiles=data["smiles"]


# Freesolv
# data=pd.read_csv("data/SAMPL.csv")
# matrix=data.iloc[:,3].values
# smiles=data["smiles"]



def smiles_to_IUPAC(smiles):
    compounds = pubchempy.get_compounds(smiles, namespace='smiles')
    match = compounds[0]
    return match.iupac_name

# store SMILES
dataset=[]
IUPAC_hm = {}
for i in range(len(smiles)):
    dataset.append([smiles[i], matrix[i]])
    IUPAC_hm[smile[i]] = smiles_to_IUPAC(smile[i])

with open('drug_data/raw/data.pkl', 'wb') as file:
    pickle.dump(dataset, file)

with open('drug_data/raw/SMILE_TO_IUPAC.pkl', 'wb') as file:
    pickle.dump(IUPAC_hm, file)

# store IUPAC

