# -*- coding: utf-8 -*-
"""
Juliana 15/05/2024


"""

import numpy as np
from pyESN import ESN 
from matplotlib import pyplot as plt
#%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

#%%
path = 'E:/caos/series/mediciones-analisis/maximos/20230925-0015_02_cruda'
d_ = np.loadtxt(path)
cortar = [50000,101000]
d=d_[cortar[0]:cortar[1]]
slices = d >= 0.01  #lugares en d que son mayores a piso                         

data= d[slices]
data/=data.max()
offset=0.2
data=data+offset
#plt.plot(data, '.-')
data2=np.copy(data) #esta copia es importante porque en el loop modifico data

trainlen = 34125

rho = 1 #spectral radius
Is = 0.9 #input_scaling
s = 0.0 #sparsity
n = 0.0003 #noise
nr=100 #n_reservoir
window=3


#IS = [0.9]#np.arange(0.1,1.5,0.1)
#S = np.arange(0,1,0.1 )
#Rho=np.arange(0.1,1.5,0.1)
#N = np.arange(0.05,0.99,0.01)
#W =np.arange(1,50,1)
Nr = [100]# np.arange(100,2000,50)
RMSE=[]

for nr in Nr:
    future = window*100
    esn = ESN(n_inputs = 1,
              n_outputs = 1,
              n_reservoir = nr,
              input_scaling=Is,
              spectral_radius = rho,
              sparsity=s,
              random_state=42,
              noise=n)
    
    
       
    pred_tot=[]
    inputs=0
    inputs_next=0
    for i in range(0, future, window):
        
        inputs = data[i : trainlen + i]
        pred_training = esn.fit(np.ones(trainlen), inputs)  # aprendizaje
        
        prediction =esn.predict(np.ones(window))
        pred_tot[i:i+window]=prediction[:,0]
    
        # Actualizar los datos de entrada eliminando los primeros 'window' puntos y agregando las predicciones realizadas
        inputs_next = np.append(inputs[window:], prediction[:, 0])
        # Actualizar 'data' con los nuevos 'inputs' para la próxima iteración
        data[i + window : trainlen + i + window] = inputs_next
    
    p=pred_tot 
    d2=data2[trainlen:trainlen+future]
    lentest=20#quiero concerntrarme en que tan bien predigo los 20 pasos siguientes a testlen
    rmse = np.sqrt(np.mean((p[:lentest] - d2[:lentest])**2))
    RMSE.append(rmse)
    print("RMSE: \n"+str(rmse)+f'  window={window}')
    
    
    plt.figure(figsize=(11,3))
    plt.plot(range(trainlen-100,trainlen+future),data2[trainlen-100:trainlen+future], 
             '.-',label="target system")
    plt.plot(range(trainlen,trainlen+future),pred_tot,'.-',
             label="free running ESN")
    lo,hi = plt.ylim()
    plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
    plt.title(f'window={window}')
    plt.legend()
    
    
#%%     
name='E:/QPLcongreso/ESNloop/barridoWindow-15052024'
parametro='Reservoir size'   

plt.figure(figsize=(11,3))
plt.plot(Nr, RMSE, '.-')
plt.title(f'RMSE en función del {parametro}: s={s}, Is={Is}, rho= {rho}, window= {window}')
plt.savefig(name,transparent=True)

arr=np.array([Nr,RMSE])
c1=f'Array {parametro},RMSE, offset={offset} Is={Is}, s={s}, rho= {rho}, noise={n},reservoir size= {nr}.'
c2=f'Path={path}, cortar={cortar}, trainlen = {trainlen}, window= {window} future={future}'

np.savetxt(name,
           arr,
           header=str(c1+c2))

