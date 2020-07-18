#--Instrucciones:--
#1. Se debe tener la imagen 'rombo.png' en la misma carpeta donde se encuentra este archivo
#2. Se debe correr, y despues de cada seccion debería aparecer un(as) imágen(es)
#3. Se de sebe apretar cualquier tecla para pasar a los siguientes resultados.

import cv2
import numpy as np
import random

import scipy.ndimage as ndi
from math import pi

from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
from matplotlib import cm

#Parte 1: abrir la imagen
img = cv2.imread('rombo.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_norm = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

noise = np.random.random(gray.shape)*0.3 #ruido gaussiano
output = gray_norm + noise

cv2.imshow('Imagen con Ruido', output)
cv2.waitKey()

#Parte 2: utilice un filtro gaussiano para reducir el ruido generado en la variable output. Para ello utilice el filtro ndi.generic_filtery los comandos apropiados para definir una máscara gaussiana.
def imfilter(A):
    A= np.reshape(A, (10,10))
    t= 10
    sigma = 2.2
    ventana= np.linspace(-t/2, t/2, t)
    u,v = np.meshgrid(ventana, ventana)
    G= (1/(sigma**2*2*pi))*np.exp(-(u**2+v**2)/(2*sigma**2))
    N= G/np.sum(G.flatten())  #normalizamos
    T = N*A
    return np.sum(T)

#aplicamos el filtro imfilter para la imagen (variable) output
img_gaussiana= ndi.generic_filter(output,imfilter, [10,10])

cv2.imshow('Filtro Gaussiano', img_gaussiana)
cv2.waitKey()

# Parte 3: determine el borde a la imagen del resultado del paso 2. 
# Utilice alguna de las técnicas vistas en clases para generar bordes. 
# Puede experimentar modificar los parámetros del paso 2.

# https://www.sicara.ai/blog/2019-03-12-edge-detection-in-opencv
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
canny = cv2.Canny(gray, 20, 30)
cv2.imshow('borde canny', canny)
cv2.waitKey(0)

# #Parte 4: utilice los algoritmos morfológicos vistos en clases para unir los bordes del paso 3. 
# #Se recomienda que realice una clausura con estructuras de distinto tamaño, es decir, 
# #que la estructura para erosionar sea menor a la dilatación con un kernel definido por usted.

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(15,15)) 
# 1 - dilate
dilate = cv2.dilate(canny, kernel)
cv2.imshow('Dilatacion', dilate)
cv2.waitKey(0)

# 2 - erosion
kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
clausura = cv2.erode(dilate,kernel2)
cv2.imshow('dilatacion + erosion',clausura)
cv2.waitKey(0)

# Parte 5: utilice la función cv2.floodFill para cerrar la región del paso 4. 
# Recuerde que dicha región debe estar cerrada para que el algoritmo rellene la región.
# https://stackoverflow.com/questions/60197665/opencv-how-to-use-floodfill-with-rgb-image
floodfill_color = 255,255,255
seed_point = 0,0
cv2.floodFill(clausura, None, seed_point, floodfill_color)
#mostrar la imagen clausura pero con el floodfill blanco por fuera del borde
cv2.imshow('floodfill',clausura)
cv2.waitKey(0)

# neg es el negativo de la imagen luego de hacer el floodfill en la imagen de clausura
neg = 255-clausura
cv2.imshow('rombo relleno en blanco',neg)
cv2.waitKey(0)
