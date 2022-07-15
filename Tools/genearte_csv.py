from itertools import count
import os
import numpy as np
import pandas as pd

# file path
big_fast_path = "./dataset/big_fast_3/"
big_slow_path = "./dataset/big_slow_3/"

# slow is 0, fast is 1
def make_dataframe(path, UAV_model, label):
    fname_list = os.listdir(path)
    label = 1 if label == 'fast'  else 0
    data = {"fname":fname_list, "directory":path, "model":UAV_model, "label":label}

    return pd.DataFrame(data)
    

fast_df = make_dataframe(big_fast_path, "X8SW", "fast")
slow_df = make_dataframe(big_slow_path, "X8SW", "slow")

data = pd.concat([fast_df, slow_df], ignore_index=True)
print(data)
data.to_csv("./information.csv")