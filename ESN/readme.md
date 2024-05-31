Programas que implementan una red tipo Echo State Network. 
Los programas son ESNloop_v0.py y ESNloop_v1.py y utilizan
redes recurrentes donde la conexión entre las neuronas es
pseudoaleatoria. Ambos programas se basan en la librería
pyESN y predicen pasos para reconstruir series temporales.
La diferencia entre ellos es que mientras que ESNloop_v0.py
usa siempre datos conocidos, ESNloop_v1.py va usando pasos
predichos para reconstruir la serie.
