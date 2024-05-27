# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:16:05 2023

@author: Juliana

Este programa contiene a la clase data_predict donde se definen dos funciones 
útiles para redes neuronales. La clase inicia con: path str con la ubicacion
del archivo serie temporal. proporcion: indica que proporcion de la serie 
se usa para train y qué proporcion para test. look_back int cuantos valores de
de la serie miro para predecir un valor y así contruir mi target. 

La función data genera los set de de datos para train y test. Devuelve los 
arrays X_train, Y_train, X_test y Y_test cada uno con su forma correspondiente.
Los X_ son filas de largo look_back, y los Y_ son los targets.

La funcion predict predice usando el modelo que se haya entrenado para predecir
multiples pasos hacia adelante.


"""

import numpy as np

class data_predic:
    
    def __init__(self,path, proporcion, look_back):
        self.path = path
        self.proporcion = proporcion
        self.look_back = look_back

    def data(self,positivo=False, cortar=[0,-1],n_features=1,std=False,cerouno=True,offset=0.):   
        '''
        positivo: hace positiva la serie, cortar corta la serie, n_features
        dimension de la serie, std normalizo por la desviación estándar, cerouno
        normalizo para que la serie vaya de cero a uno, offset sumo una 
        constante a toda la serie útil para que no haya valores muy chicos.
        
        '''
        #cargo la serie temporal
        x1=np.loadtxt(self.path)
        x=x1[cortar[0]:cortar[1]]
        #si positivo es True hago positiva la data y vuelvo a normalizar
        if positivo:
            xpositivo=(x+np.min(x))
            X=(xpositivo+abs(np.min(xpositivo)))/abs(np.max(xpositivo))

        #si positivo es false no hago nada con la data (mis series experimentales ya son positivas)
        else:
            X=x
        xs=X
        num_steps = len(xs)   
        # Indice de separacion entre train y test
        indice_train = int(self.proporcion * num_steps) 
        indice_test = indice_train + int(((1-self.proporcion)/2) * num_steps)
        #Maximo de la señal para normalizar
        maximo = np.max(np.abs(xs))
        desviacion=np.std(xs)
        # Separamos la serie en 2 partes, una para el train set y otra para el test set.
        #y normalizamos la serie si std es True se normaliza por la desviacion estandar, sino por el maximo
        
        if std:
            training_set_scaled = np.divide(xs[:indice_train:],desviacion)+offset#desde 0 hasta indice_train 
            validation_set_scaled = np.divide(xs[indice_train:indice_test:],desviacion)+offset#desde indice_train hasta indice_test
            test_set_scaled = np.divide(xs[indice_test::],desviacion)+offset#desde indice_train hasta indice_test
        else:
            if cerouno:
                xs-=xs.min()#así el minimo de xs vale cero
            training_set_scaled = np.divide(xs[:indice_train:],maximo)+offset#desde 0 hasta indice_train 
            validation_set_scaled = np.divide(xs[indice_train:indice_test:],maximo)+offset#desde indice_train hasta indice_test
            test_set_scaled = np.divide(xs[indice_test::],maximo)+offset#desde indice_train hasta indice_test
            
        X_train = []
        Y_train = []
        
        # Recorremos la serie correspondiente al train y armamos el dataset
        for i in range(self.look_back, len(training_set_scaled)):
            X_train.append(training_set_scaled[i-self.look_back:i])
            Y_train.append(training_set_scaled[i])
        X_train, Y_train = np.array(X_train), np.array(Y_train)
        
        X_val = []
        Y_val = []
        
        # Recorremos la serie correspondiente al test y armamos el dataset
        for i in range(self.look_back, len(validation_set_scaled)):
            X_val.append(validation_set_scaled[i-self.look_back:i])
            Y_val.append(validation_set_scaled[i])
        X_val, Y_val = np.array(X_val), np.array(Y_val)    
        
        X_test = []
        Y_test = []
        
        # Recorremos la serie correspondiente al test y armamos el dataset
        for i in range(self.look_back, len(test_set_scaled)):
            X_test.append(test_set_scaled[i-self.look_back:i])
            Y_test.append(test_set_scaled[i])
        X_test, Y_test = np.array(X_test), np.array(Y_test)
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
        
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    

    
    def prediccion(self,model,vec_actual,pasos_adelante):
    
        # Preparamos una lista vacia que vamos a ir llenando con los valores predichos
        lista_valores = []
        # Recorremos n=pasos_adelante pasos hacia adelante
        for i in range(pasos_adelante):    
            # Predecimos el paso siguiente
            # (El if determina si la estamos usando para la red recurrente o la densa)
            if len(vec_actual.shape)>1:
                # Prediccion Red Recurrente
                nuevo_valor = model.predict(vec_actual.reshape((1, vec_actual.shape[0], vec_actual.shape[1])))
            else:
                # Prediccion Red Densa
                nuevo_valor = model.predict(vec_actual.reshape(1, vec_actual.shape[0]))
    
            # Lo agregamos a la lista
            lista_valores.append(nuevo_valor[0][0])
    
            # Actualizmaos el vector actual con este paso
            vec_actual = np.roll(vec_actual, -1)
            vec_actual[-1] = nuevo_valor[0][0]
    
        lista_valores = np.asarray(lista_valores)
    
        return lista_valores
    
import matplotlib.pyplot as plt
   
class graficos:
  
    def learningcurve(modelmetrics,valmetrics,modelloss,valloss,i,f,name,direc):
        fig = plt.figure(figsize = (16,8))
        plt.subplot(2,1,1)
        plt.plot(modelmetrics[i:f],'.')
        plt.plot(valmetrics[i:f],'.')
        plt.title(f'model’s metrics {name}')
        plt.ylabel('mean_absolute_error')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')

        # plotting loss
        plt.subplot(2,1,2)
        plt.plot(modelloss[i:f],'.')
        plt.plot(valloss[i:f],'.')
        plt.title(f'model’s loss {name}')
        plt.ylabel('mean_square_error')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')

        plt.tight_layout()
        plt.savefig(direc)
        
    def predictioncurve(valores_reales,lista_valores,name,direc):
        fig = plt.figure(figsize = (16,8))
        plt.subplot(2,1,1)
        plt.plot(valores_reales,'.-',label='Original')
        plt.plot(lista_valores,'.-',label='Predicción')
        plt.ylabel('Serie temporal')
        plt.xlabel('Paso temporal')
        plt.title(f'Predicción de los datos de testeo {name}')
        plt.legend()
        
        plt.subplot(2,1,2)
        b=lista_valores.reshape(valores_reales.shape)
        er=(valores_reales-b)/abs(valores_reales)
        erabs=abs(er)
        plt.plot(erabs*100,'.-')
        plt.ylabel('Error absoluto [%] ')
        plt.xlabel('Paso temporal')
        plt.title(f'Error absoluto de la predicción {name}')
        
        plt.tight_layout()
        plt.savefig(direc)
        plt.show()