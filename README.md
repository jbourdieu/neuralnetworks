# neuralnetworks
En este repositorio muestro algunos programas para predicción de series temporales usando redes neuronales. 

## red_regresion_0.1.py 

Programa que crea una red neuronal profunda de capas densas utilizando la librería Keras.
(las neuronas de cada capa se conectan con todas las neuronas
la capa siguiente). 
La primer capa tiene units cantidad de neuronal y las capas
intermedias tienen 2*units cantidad de neuronas. Todas tienen 
una activación tipo relu. La última capa tiene una única 
neurona y activación tipo lineal para poder extraer la 
predicción. 


## redes_datos_prediccion.py

Programa con funciones auxiliares.
Por un lado, hay una clase que se encarga de preparar los datos
con la función data (arma los sets de train, validation y test)
y de preparar los datos para la predicción a partir de los sets
con la función prediccion. 
En la función data hay muchas opciones, esta pensada para que
acepte distintos tipos de series (le voy agregando opciones y 
cosas a medida que uso alguna serie distinta). Se puede 
simplificar. 
La función prediccion es bastante genérica, acomoda las cosas al 
formato y predice con keras. 

Por otro lado, hay una clase que arma los gráficos de las curvas
de aprendizaje y de predicción.  Acá todo es modificable a gusto
personal. 
