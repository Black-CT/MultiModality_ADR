import pandas as pd

# 读取 CSV 文件
data=pd.read_csv("data/bace.csv")
smile=data["mol"]

# 写入到 TXT 文件中
with open('../pretraining/data/train.txt', 'w') as f:
    for item in smile:
        f.write("%s\n" % item)

