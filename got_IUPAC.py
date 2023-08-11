import pickle
import pandas as pd
import pubchempy


# Use the SMILES you provided
# smiles = 'O=C(NCc1ccc(C(F)(F)F)cc1)[C@@H]1Cc2[nH]cnc2CN1Cc1ccc([N+](=O)[O-])cc1'
def smiles_to_IUPAC(smiles):
    compounds = pubchempy.get_compounds(smiles, namespace='smiles')
    match = compounds[0]
    return match.iupac_name

def store_IUPAC_HM_by_SMILES(smiles):
    compounds = pubchempy.get_compounds(smiles, namespace='smiles')
    match = compounds[0]
    print(match.iupac_name)





if __name__ == '__main__':
    data=pd.read_csv("data/Liu_data.csv")
    smiles=data["mol"]
    IUPAC_hm={}


    #test case
    # smiles = ["CC(=O)NC1=CC=C(C=C1)O"]

    for smile in smiles:
        IUPAC_hm[smile] = smiles_to_IUPAC(smile)

    with open('SMILE_TO_IUPAC.pkl', 'wb') as file:
        pickle.dump(IUPAC_hm, file)
