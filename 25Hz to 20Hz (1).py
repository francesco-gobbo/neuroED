#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os

file_folder=r'D:\Python\Temp'
file_name='H2202_A02_SW1_SAM.csv'

DLC_file= os.path.join(file_folder, file_name)

df=pd.read_csv(DLC_file, sep=',', index_col=0)
xy=df[['x','y']]


# In[ ]:


a=0

df_25_Hz = pd.DataFrame(columns=['x', 'y', 'time'])
df_20_Hz = pd.DataFrame(columns=['x', 'y', 'time'])

x=df['x']
y=df['y']
                  
for i in xy.index:
    df_25_Hz = df_25_Hz.append({'x': x[i], 'y': y[i], 'time': i*40}, ignore_index=True)
    
df_25_Hz


# In[ ]:


df_20_Hz = pd.DataFrame(columns=['x', 'y', 'time'])


i = 0
j=0
while i <= (len(df.index)-4):
    df_20_Hz = df_20_Hz.append({'x': x[i], 'y': y[i], 'time': j*50}, ignore_index=True)
    j+=1
    df_20_Hz = df_20_Hz.append({'x': (3*x[i+1] + x[i+2])/4, 'y': (3*y[i+1] + y[i+2])/4, 'time': j*50}, ignore_index=True)
    j+=1
    df_20_Hz = df_20_Hz.append({'x': (x[i+2] + x[i+3])/2, 'y': (y[i+2] + y[i+3])/2, 'time': j*50}, ignore_index=True)
    j+=1
    df_20_Hz = df_20_Hz.append({'x': (x[i+3] + 3*x[i+4])/4, 'y': (y[i+3] + 3*y[i+4])/4, 'time': j*50}, ignore_index=True)
    j+=1
    i=i+5


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(df['x'], df['y'], c= 'orange')
plt.plot(df_20_Hz['x'], df_20_Hz['y'], color = 'grey')


df_20_Hz


# In[ ]:


df_20_Hz.loc[df_20_Hz['time'] > 100000]


# In[ ]:


df_25_Hz.loc[df_25_Hz['time'] > 100000]


# In[ ]:




