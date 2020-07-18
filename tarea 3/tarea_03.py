#--Instrucciones:--
#1. Se debe tener la imagen 'fallas.tif' en la misma carpeta donde se encuentra este archivo
#2. Se debe correr, y despues de cada seccion debería aparecer un(as) imágen(es)
#3. Se de sebe apretar cualquier tecla para pasar a los siguientes resultados.

import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

#Parte 1
#1 - Lectura de un imagen
img = cv2.imread('claudio_carla.jpeg')

#2 - Agregar ruido frecuencias vertical
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_img = cv2.normalize(gray_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

cv2.imshow('Imagen Gris', gray_img)

cv2.waitKey()

#vamos a crear ruido
#10Hz
V = np.linspace(0,1,num=512)
Y= 0.2*np.sin(2*np.pi*10*V) 
M= np.matlib.repmat(Y,512,1)
noise_img_10= np.add(M, gray_img)
cv2.imshow('Imagen con Ruido Frecuencial de 10 Hz', noise_img_10)

cv2.waitKey()

#30Hz
Y= 0.2*np.sin(2*np.pi*30*V) 
M= np.matlib.repmat(Y,512,1)
noise_img_30= np.add(M, gray_img)
cv2.imshow('Imagen con Ruido Frecuencial de 30 Hz', noise_img_30)

cv2.waitKey()

#50Hz
Y= 0.2*np.sin(2*np.pi*50*V) 
M= np.matlib.repmat(Y,512,1)
noise_img_50= np.add(M, gray_img)
cv2.imshow('Imagen con Ruido Frecuencial de 50 Hz', noise_img_50)

cv2.waitKey()

#3 - Borrar ruido con filtros de Fourier
#vamos a aplicar fourier a imagen de 10 Hz
F = np.fft.fft2(noise_img_10)
fshift = np.fft.fftshift(F)

spectrum = 0.1*np.log(np.abs(fshift))
cv2.imshow('Espectro con ruido de Imagen con Ruido 10Hz', spectrum) #espectro de la imagen F_10 (imagen con ruido)

#borrando los puntos blancos
fshift[256:258, 246:248]=0.0
fshift[256:258, 266:268]=0.0

S= np.fft.ifft2(np.fft.fftshift(fshift))
S= S.real

spectrum = 0.1*np.log(np.abs(fshift))
cv2.imshow('Espectro arreglado de Imagen con Ruido 10Hz', spectrum) #espectro de la imagen F_10 borrando los dos puntos blanco

cv2.imshow('Output de Imagen con Ruido 10Hz - Filtro Fourier', S)

cv2.waitKey()

#vamos a aplicar fourier a imagen de 30 Hz
F = np.fft.fft2(noise_img_30)
fshift = np.fft.fftshift(F)

spectrum = 0.1*np.log(np.abs(fshift))
cv2.imshow('Espectro con ruido de Imagen con Ruido 30Hz', spectrum) #espectro de la imagen F_10 (imagen con ruido)

#borrando los puntos blancos
fshift[256:258, 224:230]=0.0
fshift[256:258, 284:290]=0.0

S= np.fft.ifft2(np.fft.fftshift(fshift))
S= S.real

spectrum = 0.1*np.log(np.abs(fshift))
cv2.imshow('Espectro arreglado de Imagen con Ruido 30Hz', spectrum) #espectro de la imagen F_10 borrando los dos puntos blanco

cv2.imshow('Output de Imagen con Ruido 30Hz - Filtro Fourier', S)

cv2.waitKey()

#vamos a aplicar fourier a imagen de 50 Hz
F = np.fft.fft2(noise_img_50)
fshift = np.fft.fftshift(F)

spectrum = 0.1*np.log(np.abs(fshift))
cv2.imshow('Espectro con ruido de Imagen con Ruido 50Hz', spectrum) #espectro de la imagen F_10 (imagen con ruido)

#borrando los puntos blancos
fshift[256:258, 202:212]=0.0
fshift[256:258, 302:312]=0.0

S= np.fft.ifft2(np.fft.fftshift(fshift))
S= S.real

spectrum = 0.1*np.log(np.abs(fshift))
cv2.imshow('Espectro arreglado de Imagen con Ruido 50Hz', spectrum) #espectro de la imagen F_10 borrando los dos puntos blanco

cv2.imshow('Output de Imagen con Ruido 50Hz - Filtro Fourier', S)

cv2.waitKey()

#Parte 2
#1 - Genere una imagen con ruido empleando el siguiente código. 
img = cv2.imread('cameraman.png',cv2.IMREAD_GRAYSCALE)
m = img.shape[0]
delta=15 
V=np.fix(np.linspace(delta,m-delta,delta)).astype('uint8')

img[V,:]=img[V,:]+50 
img[:,V]=img[:,V]+50 

cv2.imshow('ruido', img) 

cv2.waitKey()

#2 - Luego utilice un filtro en la frecuencia para reducir dicho ruido. 
#Usted debe definir el filtro más apropiado.

#Usaremos el filtro Gaussiano ya que este es un filtro pasabajo, 
#esto permitirá mantener la idea general de la imagen, 
#difimunando el ruido que es más detallado. 

                #tamaño de la imagen
x= np.linspace(-127, 128,256)
y= np.linspace(-127, 128,256)
X, Y = np.meshgrid(x,y)

#creamos el filtro
H= np.exp(-0.01*(np.power(X,2)+np.power(Y,2)));
UH = np.fft.fftshift(H)

F = np.fft.fft2(img)

FILT= UH*F

S= np.fft.ifft2(np.fft.fftshift(FILT))
out = cv2.normalize(abs(S), None, 0.0, 1.0, cv2.NORM_MINMAX)

cv2.imshow('Imagen con Ruido Reducido Mediante Filtro Gaussiano', out)
cv2.waitKey()