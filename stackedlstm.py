
"""

Programa que implementa una red de capas LSTM apiladas para la predicción de 
series temporales. 
El programa está diseñado para que se pueda realizar un barrido en cantidad de
épocas, cantidad de capas apiladas y tamaño del batch. 

 
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import  keras.callbacks
#from keras import initializers #útil si quiero configurar los pesos iniciales
 
from IPython import get_ipython
#import matplotlib.pyplot as plt
from redes_datos_prediccion import data_predic, graficos
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'qt')



########## Configuración de los datos ##########

path='E:/caos/06112023/maximos/20231106-0004_07_cruda' #direccion de los datos


proporcion = 0.7 #proporcion de la serie que se utiliza para entrenar
look_back = 10 #cada cuantos valores que aprendo tengo un target
cortar=[0,10000] #si quiero cortar la serie 
n_features=1 #dimensión de la serie a predecir

dp=data_predic(path, proporcion, look_back)  #inicio el modulo data_predict
#creo los set de datos para train, validación y testeo
X_train, Y_train, X_val, Y_val ,X_test, Y_test = dp.data(offset=0.1,positivo=False,
                                                         cortar=cortar,
                                                         n_features=1,std=False,
                                                         cerouno=True)

################################################


########### Configuración del modelo ###########
    
Epochs=[100]#cantidad de epocas máximas a entrenar, lista o array
batch_num = [150]#tamaño del batch, lista o array 
unitslstm=150#cantidad de neuronas de las capas LSTM 
stack=[5]#cantidad de capas a apilar, lista o array de enteros mayores o iguales a 1
r_drop=0.#recurrent dropout de las capaz LSTM: proporcion, valor entre 0 y 1 (0: dropout desactivado)
d_drop=0.#dropout para la capa densa de activacion: proporcion, valor entre 0 y 1 (0: dropout desactivado)

#estas lineas comeentados son ejemplos por si quiero configurar los pesos de inicializacion del modelo
#weights=[np.random.random([n_features,4*unitslstm]),np.random.random([unitslstm,4*unitslstm]),np.random.random([4*unitslstm,])]
#weights=[np.ones([1,400]),np.ones([100,400]),np.ones([400,])]


#defino el tipo de modelo
def buildmodel(stack):
    # Defino el modelo Secuencial
    model = Sequential()
    # Capas LSTM
    #distingo entre la posibilidad de tener 1, 2 o mas capas apiladas
    
    if stack==1: 
        model.add(LSTM(unitslstm, activation='tanh',return_sequences=False,
                       input_shape=(look_back, n_features),
                       recurrent_dropout=r_drop))
        model.add(Dropout(d_drop))
        #Capa de salida
        model.add(Dense(1,activation='linear'))
        # Compilo
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                      metrics=['mean_absolute_error'], loss='mse')
        
    if stack==2:
        model.add(LSTM(unitslstm, activation='tanh',return_sequences=True,
                       input_shape=(look_back, n_features),
                       recurrent_dropout=r_drop))
        model.add(LSTM(unitslstm, activation='tanh',return_sequences=False,
                       recurrent_dropout=r_drop))
        #capas densas
            
        model.add(Dropout(d_drop))
        #Capa de salida
        model.add(Dense(1,activation='linear'))
        # Compilo
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                      metrics=['mean_absolute_error'], loss='mse')
        
    if stack>2:
        model.add(LSTM(unitslstm, activation='tanh',return_sequences=True,
                       input_shape=(look_back, n_features),
                       recurrent_dropout=r_drop))
        for i in range(stack-2):
            model.add(LSTM(unitslstm, activation='tanh',return_sequences=True,
                           recurrent_dropout=r_drop))
        
        model.add(LSTM(unitslstm, activation='tanh',return_sequences=False,
                       recurrent_dropout=r_drop))
            
        model.add(Dropout(d_drop))
        #Capa de salida
        model.add(Dense(1,activation='linear'))
        # Compilo
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
                      metrics=['mean_absolute_error'], loss='mse')
        
    return model
#################################################



############### Compilo el modelo ###############
'''
observar que la compilación del modelo está pensada para que se pueda realizar
un barrido en cantidad de capas a apilar, cantidad de epocas máximas y tamaño 
del batch
'''

for s in stack:    
    model=buildmodel(s)
    #model.layers[0].set_weights(weights)
    
    #Defino la lista de callbacks. ModelCheckpoint va guardando las epocas del modelo
    for e in Epochs:  
        for b in batch_num:
            
            time=datetime.now().strftime('%Y_%m_%d_%H%M')
            modelname='C:/Users/Solidos/Desktop/stackedlstm_'+time #direccion para guardar el modelo
    
            print('\n'+modelname+'\n')
            #lista de callbacks: c1 para la compilacion en la mejor epoca y deja los mejores pesos
            #c2: guarda el mejor modelo
            c1=keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, 
                                             baseline=0.01, restore_best_weights=True,
                                             start_from_epoch=50)
            c2= keras.callbacks.ModelCheckpoint(filepath=modelname+'.keras',
                                                monitor="val_loss", save_best_only=True)
            callbacks_list = [c1 ,c2]
                
            # Entreno el modelo y lo guardo en history
            history = model.fit(X_train, Y_train, batch_size = b, epochs=e,
                                callbacks=callbacks_list, verbose=1,
                                validation_data=(X_val, Y_val))            
            #cargo el modelo compilado y lo evalúo
            loaded_model = keras.saving.load_model(f"{modelname}.keras")
            results=loaded_model.evaluate(x = X_test, y = Y_test, batch_size=b)
            print("test loss, test metrics:", round(results[0],4),round(results[1],4))
            
            #guardo la metrica y la funtion loss
            modelmetrics = history.history['mean_absolute_error']
            valmetrics = history.history['val_mean_absolute_error']
            modelloss = history.history['loss']
            valloss = history.history['val_loss']
            
            x=np.array([modelmetrics, valmetrics, modelloss, valloss])            
            com1 = f'mean_absolute_error,val_mean_absolute_error,loss,val_loss. Serie {path}, corte {cortar},'
            com2 = f'test loss, test metrics: {round(results[0],4)},{round(results[1],4)}'
            comentario = com1 + com2
            np.savetxt(f'{modelname}_history',x,header=comentario)
            
            # grafico las curvas de aprendizaje
            nombre1 = 'stackedlstm_'+time+f'stack,epoch,batch:{s}, {e}, {b}'
            graficos.learningcurve(modelmetrics,valmetrics,modelloss,valloss,
                                   i=0,f=-1,name= nombre1,
                                   direc=f'{modelname}_learning')
    
            ###### Predicción ######
            pasos_adelante = 10 #cantidad de pasos a predecir
            indice_inicial = 0 #índice en el set de testeo desde el cual comienzo la predicción
            
            
            # tomo un valor inicial del test set
            vec_actual = X_test[indice_inicial]
            
            # calculo la prediccion del modelo
            lista_valores = dp.prediccion(loaded_model, vec_actual, pasos_adelante)
            
            # tomo los valores esperados
            valores_reales = Y_test[indice_inicial:indice_inicial+pasos_adelante]
            
            # grafico la prediccion
            nombre2 = 'stackedlstm_'+time+f'stack,epoch,batch:{s}, {e}, {b}'
            graficos.predictioncurve(valores_reales,lista_valores,
                                     name=nombre2,direc=f'{modelname}_predic')