import pickle

if __name__ == '__main__':
    with open("../raw/SMILE_TO_IUPAC.pkl", "rb") as f:
        data = pickle.load(f)
