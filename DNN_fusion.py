import numpy as np
import datetime as dt
import glob as gl
import os
import warnings
import netCDF4 as nc4
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from sklearn.preprocessing import StandardScaler, PowerTransformer
import joblib

gridX = np.arange(70+0.025,150+0.025,0.05, dtype=np.float32)
gridY = np.arange(-10+0.025,50+0.025,0.05, dtype=np.float32)
gridX, gridY = np.meshgrid(gridX, gridY)

fusion_date = start_date
while fusion_date<end_date:
    print(fusion_date)

    fused_AOD = np.full(gridX.shape, np.nan)
    fused_QF = np.full(gridX.shape, -1)
    for flag in range(6):
        with open(fusion_date.strftime(f'test_dataset_%Y%m%d%H%M_flag={flag:01}.pkl'), 'rb') as fr:
            inputs = pk.load(fr)

            if flag == 0:   n_aod = 4
            elif flag in [1,3]:    n_aod = 3
            elif flag in [2,5]:    n_aod = 2
            else:n_aod = 1
            inputs[:,:n_aod] += 0.05

            scaler_in_aod = joblib.load(f'flag={flag:1d}_scaler_in_aod.sav') 
            scaler_in_other = joblib.load(f'flag={flag:1d}_scaler_in_other.sav') 
            if flag in [0,1,2,3,5]:
                inputs[:,:n_aod] = scaler_in_aod.transform(inputs[:,:n_aod])
                inputs[:,n_aod:] = scaler_in_other.transform(inputs[:,n_aod:])
            else:
                inputs[:,:n_aod] = scaler_in_aod.transform(inputs[:,:n_aod].reshape(-1,1))
                inputs[:,n_aod:] = scaler_in_other.transform(inputs[:,n_aod:])

            # #########################################################################
            # Numpy array >> Tensor
            # #########################################################################
            class CustomDataset(Dataset):
                def __init__(self):
                    # self.labels = output
                    self.features = inputs

                def __len__(self):
                    return len(self.labels)

                def __getitem__(self, idx):
                    x = torch.FloatTensor(self.features[idx])
                    return x

            dataset = CustomDataset()

            # #########################################################################
            # Prediction
            # #########################################################################
            model = torch.jit.load(f'flag={flag:1d}_fusion_model.pth')
            scaler_out = joblib.load(f'flag={flag:1d}_scaler_out.sav') 

            with torch.no_grad():
                model.eval()
                # X = dataset
                X = torch.FloatTensor(inputs)
                X = X.to(device)
            
                pred = model(X)
                pred = pred.numpy()
                pred = scaler_out.inverse_transform(pred)

            fused_AOD[idx] = pred.reshape(-1)
            fused_QF[idx] = flag

    nc = nc4.Dataset(fusion_save_dir+fusion_date.strftime('/%Y%m/GK2_L4_DNN_%Y%m%d%H%M_AOD-FUSION.nc'), 'w', format='NETCDF4')
    nc.createDimension('lat', gridX.shape[0])
    nc.createDimension('lon', gridX.shape[1])
    lat_nc = nc.createVariable('lat', np.float32, ('lat','lon'))
    lat_nc.standard_name = 'latitude'; lat_nc.long_name = 'latitude'
    lat_nc.units = 'degrees_north'; lat_nc.axis = 'Y'
    lat_nc[:,:] = gridY
    lon_nc = nc.createVariable('lon', np.float32, ('lat','lon'))
    lon_nc.standard_name = 'longitude'; lon_nc.long_name = 'longitude'
    lon_nc.units = 'degrees_east' ; lon_nc.axis = 'X'
    lon_nc[:,:] = gridX
    write_nc(nc, fused_AOD, 'fused AOD 550nm', 'GEMS V2+AMI(YAER)+GOCI-II fused Aerosol Optical Depth at 550nm', '', [0,65534], dtype=np.uint16)
    write_nc(nc, fused_QF, 'QF', '0=all 1=GEMS+AMI 2=GEMS+GOCI 3=AMI+GOCI 4=GEMS 5=AMI 6=GOCI', '', [0,6], dtype=np.int8)
    nc.close()

    fusion_date+=dt.timedelta(hours=1)
    if fusion_date.hour == 8: fusion_date+=dt.timedelta(hours=16)
