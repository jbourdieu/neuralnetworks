# -*- coding: utf-8 -*-
"""
Programa para predecir series temporales usando una red neuronal tipo 
feedforward de capas densas. 

"""


from keras.models import Sequential
from keras.layers import Dense, Dropout
import  keras.callbacks

import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

from datetime import datetime
from redes_datos_prediccion import data_predic, graficos


time=datetime.now().strftime('%Y_%m_%d_%H%M')
modelname='modelo_'+time

########## PARÁMETROS DE DATOS Y GENERACION DE SETS ##########

path='E:/caos/23102023/maximos/20231023-0002_1_cruda'
proporcion = 0.7 #proporcion de la serie que uso para entrenar
look_back = 50 #cada cuantos valores que aprendo tengo un target
cortar=[0,10000] #si no queiero usar la serie entera
n_features=1 #dimensión de lo que quiero predecir (en este caso las series pertenecen a R)

#GENERO LOS SET DE DATOS
dp=data_predic(path, proporcion, look_back)
X_train, Y_train, X_val, Y_val ,X_test, Y_test = dp.data(offset=0.1,positivo=False,
                                                         cortar=cortar,
                                                         n_features=1,
                                                         std=False,
                                                         cerouno=True)


########## PARÁMETROS DE DATOS Y GENERACION DE SETS ##########

########## PARÁMETROS DEL MODELO ##########

units=200 #cantidad de neuronas de cada capa
hidden=20 #cantidad de capas entre la primera y la última
#do=0.0 #dropout proporcón de neuronas que se apagan = 0

###########################################

########## PARÁMETROS DE ENTRENAMIENTO ##########

batch=100 #tamaño del batch
Epochs=200 #cantidad de épocas a entrenar

#################################################



########## MODELO ##########

model = Sequential()
model.add(Dense(units, input_shape=(look_back,)))#capa inicial
#model.add(Dropout(do))

for i in range(0,hidden):
    model.add(Dense(2*units, activation='relu'))#activation puede ser tanh
    #model.add(Dropout(do))
model.add(Dense(1, activation='linear'))

# compiling the sequential model
model.compile(loss='mse', metrics=['mean_absolute_error'], optimizer='adam')

############################

########## CALLBACKS ##########

c1 = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, 
                                 baseline=0.0001, restore_best_weights=True,
                                 start_from_epoch=50)
c2= keras.callbacks.ModelCheckpoint(filepath=modelname+'.keras',
                                    monitor="val_loss", save_best_only=True)
callbacks_list = [c1 ,c2]

###############################



########## ENTRENAMIENTO ##########
# training the model and saving metrics in history
history = model.fit(X_train, Y_train, callbacks=callbacks_list,
          batch_size=batch, epochs=Epochs,
          verbose=1,
          validation_data=(X_val, Y_val))

loaded_model = keras.saving.load_model(f"{modelname}.keras") #cargo el modelo que mejor resultó
#Acá uso el test para evaluar el error del modelo YA ENTRENADO
results=loaded_model.evaluate(x = X_test, y = Y_test, batch_size=batch) #evaluo el modelo usando el test set

print("test loss (MSE), test metrics (MAE):", round(results[0],4),round(results[1],4))

###################################

########## GUARDAR HISTORY (OPC)  ##########
#guardo la metrica y la funtion loss
modelmetrics = history.history['mean_absolute_error']
valmetrics = history.history['val_mean_absolute_error']
modelloss = history.history['loss']
valloss = history.history['val_loss']

x = np.array([modelmetrics, valmetrics, modelloss, valloss])
com1 = 'array: mean_absolute_error,val_mean_absolute_error,loss,val_loss.'
com2 = f' Serie {path}, corte {cortar}'
com3 = f' test loss, test metrics: {round(results[0],4)},{round(results[1],4)}'
comentario = com1 + com2 + com3
np.savetxt(f'{modelname}_history',x,header=comentario)

############################################


########## LEARNING CURVES ##########
#evolucion temporal del modelo, comparo train set con validation set
graficos.learningcurve(modelmetrics,valmetrics,modelloss,valloss,i=0,f=-1,
                       name='stackedlstm_'+time+f'batch:{batch}',
                       direc=f'{modelname}_learning')


#%%
########## PREDICCIÓN ##########


pasos_adelante = 80
indice_inicial = 0

# Tomamos un valor inicial del test set
vec_actual = X_test[indice_inicial]

# Calculamos la prediccion del modelo. Obs: para esto uso el test set
lista_valores = dp.prediccion(loaded_model, vec_actual, pasos_adelante)

# Tomamos los valores esperados
valores_reales = Y_test[indice_inicial:indice_inicial+pasos_adelante]

# grafico la prediccion
graficos.predictioncurve(valores_reales,lista_valores,
                         name='stackedlstm_'+time+f'batch: {batch}',
                         direc=f'{modelname}_predic')

################################
