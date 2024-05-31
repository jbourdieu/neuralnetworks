# -*- coding: utf-8 -*-
"""
Juliana 15/05/2024

Programa que implementa un ESN para predecir una serie temporal. En este caso 
se predicen window pasas hacia adelante hasta llegar a future cantidad de pasos.
Cuando predigo los pasos trainlen + 0 y trainlen + 1 uso el set completo de 
entrenamiento train_set[0:trainlen]. Luego, guardo la predicción en pred_tot, 
quito los primeros dos puntos de mi train set y agrego al train set los puntos
correspondientes a lugares trainlen + 0 y trainlen + 1 de la serie. O sea que, 
en cada window  actualizo train_set por los puntos siguientes en la serie 
temporal.

Si bien este tipo de modelo aparenta predecir muy bien, tener en 
cuenta que, debido a la forma en la que se actualiza el train, la capacidad de 
predicción es solamente window pasos.

"""

import numpy as np
from pyESN import ESN 
from matplotlib import pyplot as plt
#%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

d = np.loadtxt('E:/caos/06112023/maximos/20231106-0004_07_cruda')
#slices = d >= 0.02  #lugares en voltaje que son mayores a piso                         

#data= d[slices]
#data/=data.max()
#data=data+0.2
data=d

trainlen = 15000
#trainlen = 10000

window=10
future = window*10

pred_tot=np.zeros(future)
Rho=[1.25]#np.arange(0.25,1.5,0.25)

#obs que acá primero hago un loop en rho. Esto sirve si quiero optimizar este 
#parámetro, sino dejar RHO como un array o lista de un único elemento
for rho in Rho:
    esn = ESN(n_inputs = 1,
              n_outputs = 1,
              n_reservoir = 159,
              spectral_radius = rho,
              sparsity=0.0,
              random_state=42)
    for i in range(0,future,window):
        inputs=data[i:trainlen+i]
        pred_training = esn.fit(np.ones(trainlen),inputs) #aprendizaje
    
        prediction = esn.predict(np.ones(window))
        pred_tot[i:i+window]=prediction[:,0]
        
        
    print("test error: \n"+str(np.sqrt(np.mean((pred_tot - data[trainlen:trainlen+future])**2))))
    
    plt.figure(figsize=(11,3))
    plt.plot(range(trainlen-200,trainlen+future),data[trainlen-200:trainlen+future], 
             '.-',label="target system")
    plt.plot(range(trainlen,trainlen+future),pred_tot,'.-',
             label="free running ESN")
    lo,hi = plt.ylim()
    plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
    plt.legend()
