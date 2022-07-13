from itertools import count
import os
import numpy as np
import pandas as pd

# define file path
big_fast_path = "./dataset_3sec/big_fast/"
big_slow_path = "./dataset_3sec/big_slow/"
small_fast_path = "./dataset_3sec/small_fast/"
small_slow_path = "./dataset_3sec/small_slow/"

# slow is 0, fast is 1
def make_dataframe(path, UAV_model, label):
    fname_list = os.listdir(path)
    label = 1 if label == 'fast'  else 0
    data = {"fname":fname_list, "directory":path, "model":UAV_model, "label":label}

    return pd.DataFrame(data)
    
# make dataframe for concat
# name_of_define = make_dataframe(file_path, "name_of_model", "fast or slow")
b_fast_df = make_dataframe(big_fast_path, "X8SW", "fast")
b_slow_df = make_dataframe(big_slow_path, "X8SW", "slow")
s_fast_df = make_dataframe(small_fast_path, "X5UW", "fast")
s_slow_df = make_dataframe(small_slow_path, "X5UW", "slow")

# use pandas.concat to combine the data
b_data = pd.concat([b_fast_df, b_slow_df], ignore_index=True)
s_data = pd.concat([s_fast_df, s_slow_df], ignore_index=True)
data = pd.concat([b_data, s_data], ignore_index=True)

print(data)
data.to_csv("./information.csv")
