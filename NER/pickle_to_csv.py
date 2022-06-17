import pickle as pkl
import pandas as pd
with open("/Users/douglas-adams/Downloads/pages_list_nodes", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv(r'ER_pages.csv')