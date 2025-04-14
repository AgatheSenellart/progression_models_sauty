import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

path = '/home/asenella/data/starmen/output_random'
dict_data = {'data' : [], 'timepoints' : [], 'labels': []}

df = pd.read_csv(os.path.join(path, 'df.csv'))

for i in tqdm(range(1000)):

    for t in range(10):

        file_id = f'SimulatedData__Reconstruction__starman__subject_s{i}__tp_{t}.npy'
        
        # get the image
        dict_data['data'].append(torch.from_numpy(np.load(os.path.join(path + '/images', file_id))).unsqueeze(0))

        # get the associated time
        dict_data['timepoints'].append(float(df['t'][df['path']==file_id].values[0]))

        dict_data['labels'].append(i)

dict_data['data'] = torch.stack(dict_data['data'])
dict_data['timepoints'] = torch.Tensor(dict_data['timepoints'])
dict_data['labels'] = torch.Tensor(dict_data['labels'])

torch.save(dict_data,os.path.join(path, 'dict_data_sauty.pt'))