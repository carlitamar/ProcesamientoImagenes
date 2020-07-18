#--Instrucciones:--
#1. Se debe tener la imagen 'gordis.png' y 'cameraman.png' en la misma carpeta donde se encuentra este archivo
#2. Se debe correr, y despues de cada seccion debería aparecer un(as) imágen(es)
#3. Se de sebe apretar cualquier tecla para pasar a los siguientes resultados.

import cv2
import numpy as np
import random
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, draw, pause
from matplotlib import cm
import scipy.ndimage as ndi

#Parte 1A: Genere distintas versiones de la imagen original empleando distintos modelos de ruido.
#- Lectura imagen 'gordis.png'
img_gordis = cv2.imread('gordis.png')
gray = cv2.cvtColor(img_gordis,cv2.COLOR_BGR2GRAY)
gray_norm = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

cv2.imshow('Imagen Original', gray_norm)
cv2.waitKey()

#1 - Ruido Gaussiano
noise = np.random.random(gray.shape)*0.35
gaussiano = gray_norm + noise

cv2.imshow('Ruido Gaussiano', gaussiano)
cv2.waitKey()

#2 - Ruido Uniforme
#se genera una matriz del tamaño shape (del tamaño de la imagen) 
uniform_noise = np.zeros((gray.shape[0], gray.shape[1]),dtype=np.uint8)
#con numeros aleatorios de ruido (se suman al numero del pixel normal)
cv2.randu(uniform_noise,0,120)

uniforme = cv2.add(gray,uniform_noise)

cv2.imshow('Ruido Uniforme',uniforme)
cv2.waitKey()

#3 - Ruido Impulsional Sal (white)
def filtro_promedio_podado(A):
    d=3
    S= np.sort(A.flatten())
    B= S[d:-d]
    L= len(B)
    val=(1.0/L)*np.sum(B)
    return val

mat_noise_white=np.random.random(gray.shape); #creates a uniform random variable from 0 to 1 
sp_noise_white= np.uint8(np.where(mat_noise_white>=0.85, 255,0))
noise_img_white = cv2.add(gray,sp_noise_white)
sal= ndi.generic_filter(noise_img_white,filtro_promedio_podado, [3,3])

cv2.imshow('Ruido Impulsional Sal', sal)
cv2.waitKey()

#4 - Ruido Impulsional Pimienta (black)
mat_noise_black=np.random.random(gray.shape); #creates a uniform random variable from 0 to 1 
sp_noise_black= np.uint8(np.where(mat_noise_black>=0.15,  1,0))
noise_img_black = cv2.multiply(gray,sp_noise_black)
pimienta= ndi.generic_filter(noise_img_black,filtro_promedio_podado, [3,3])

cv2.imshow('Ruido Impulsional Pimienta', pimienta)
cv2.waitKey()

#Parte 1B: Por cada imagen ruidosa determine el filtro que mejor reduzca ruido. Utilice sólo un filtro por cada imagen. 
#No obstante, el filtro que usted seleccione debe pertenecer a una de las tres familias de <<filtros en el espacio>>. 
#Esto significa que al menos debe haber un filtro de <<orden estadístico>>, un <<filtro adaptivo>>, y un <<filtro lineal>> implementado en su tarea. 
#Explique las razones por las cuales el filtro que usted seleccionó es mejor que los otros filtros.

#1 - Filtro para Ruido Gaussiano
#Orden Adaptativo-Ruido Local: Este filtro determina medidas estadísticas simples como la media y la varianza en la región a analizar.
#El ruido gaussiano se puede reducir más eficientemente mediante el uso de filtros frecuenciales, pero debido al desafío de la tarea, se debió hacer mediante un filtro espacial. Mirando los filtros a nuestra disposición, la mayoría servían mejor en ruido impulsional, pero este era bueno para ruido gaussiano siempre y cuando se  se ajusta bien la varianza del ruido (variable var_N del código) y si el ruido no es muy elevado. Si elevamos  la varianza del ruido, nos empeora la imagen y si disminuimos mucho la varianza, no nos arregla el ruido. Para nuestro caso elegimos = 0,01.
def filtro_ruido_local(A):
    var_N = 0.01
    B = A.flatten()
    n = len(B)
    #calcula la varianza de B
    var_L = np.var(B)
    mu    = np.mean(B)
    #dame el valor de la posicion central de tu lista (no ordenados, asique no es mediano)
    #uint8(n/2) > trunca el valor para que de uno entero
    g     = B[np.uint8(n/2)]     
    f     =  g-(var_N/var_L)*(g-mu)
    return f 

filtro_ruidolocal= ndi.generic_filter(gaussiano,filtro_ruido_local, [3,3])
cv2.imshow('Filtro Adaptativo-Ruido-Local para Ruido Gaussiano', filtro_ruidolocal)
cv2.waitKey()

#2 - Filtro para Ruido Uniforme
#Orden Lineal-Media: Reemplaza el valor central de la máscara por el promedio de los valores contenidos en ella.
#El ruido uniforme es de los ruido más difíciles de disminuir, ya que afecta toda la imagen de manera uniforme y al menos que sea de tipo frecuencial, no se podrá disminuir de esa manera. En este caso, no usamos un ruido uniforme frecuencial. Como ya dijimos anteriormente, la mayoría servían mejor en ruido impulsional, pero este era bueno para ruido uniforme, ya que elimina el ruido reemplazandolo por el promedio de la máscara. Lo que podemos variar es el tamaño de la máscara y mientras más grande sea más borroso se verá la imágen. En este caso, usamos una máscara de 3x3.
def filtro_media(A):
    S= np.mean(A.flatten())    
    return S

filtro_media= ndi.generic_filter(uniforme,filtro_media, [3,3])
cv2.imshow('Filtro Estadistico-Media para Ruido Uniforme', filtro_media)
cv2.waitKey()

#3 - Filtro para Ruido Impulsional Sal
#Orden Estádístico-Min: Reemplaza el valor central de la máscara por el valor mínimo de la máscara. 
#El ruido sal es fácil de disminuir mediante el uso del filtro mínimo, ya que reemplaza el ruido por el valor mínimo de la máscara. Es decir, que siempre va a poder eliminar ese valor ruido (muy alto/blanco) por un valor más adecuado (el mínimo de la máscara) para "colorear" ese espacio en blanco.
def filtro_min(A):
    S= np.min(A.flatten())    
    return S

filtro_sal= ndi.generic_filter(sal,filtro_min, [3,3])
cv2.imshow('Filtro Estatico-Min para Ruido Impulsional Sal', filtro_sal)
cv2.waitKey()

#4 - Filtro para Ruido Impulsional Pimienta
#Orden Lineal-Contra Armónica: Reemplaza el valor central de la máscara por la media contra armónica.
#El filtro contra-armónico reemplaza el valor del ruido por la media-contra armónica. Siempre va a poder eliminar ese valor ruido (muy bajo/negro) por un valor más adecuado (media contra-armónica) para "colorear" ese espacio en negro.
def filtro_contra_armonico(A, Q): #puedes dar parametros adiciones con extra_keywords y aca lo defines para usarlo
    S= np.sum(A.flatten()**(Q+1)) / np.sum(A.flatten()**Q)
    return S

#mientras mas alto Q, más claro se pone
filtro_pimienta= ndi.generic_filter(pimienta,filtro_contra_armonico, [3,3], extra_keywords={'Q':5})
cv2.imshow('Filtro Lineal-Contra-Armonica para Ruido Impulsional Pimienta', filtro_pimienta)
cv2.waitKey()

# ----------------------------------------------------------------------------
# Parte 2: Restaure la imagen del siguiente código empleando el filtro de Wiener y el filtro Paramétrico. 
# Determine el parámetro que mejor resultado genere.
img= cv2.imread('cameraman.png', cv2.IMREAD_GRAYSCALE)

gray = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) 
F = np.fft.fft2(gray)
vector = np.linspace(-0.5, 0.5, gray.shape[0])
U,V = np.meshgrid(vector, vector)

a = 5; b = 1
UV = U*a+V*b

H = np.fft.fftshift(np.sinc(np.pi*UV)*np.exp(-1j*np.pi*UV)) 
G = H*F
g = np.real(np.fft.ifft2(G))

cv2.imshow('G', g)
cv2.waitKey()

#filtro de Wiener
#valor de k se puede cambiar
#W = np.conj(H)/(np.abs(H)**2 + 0.1) #los detalles se ven menos nítidos pero el efecto de fenómeno es menor (el fenómeno de gibbs tmb genera ruido)
W = np.conj(H)/(np.abs(H)**2 + 0.0006) #los detalles se ven más nítidos pero el efecto de fenómeno es mayor
G = np.fft.fft2(g)
F = W*G
iRestored_W = np.real(np.fft.ifft2(F))

cv2.imshow('Imagen Restaurada con Filtro de Wiener', iRestored_W)
cv2.waitKey()

#filtro Paramétrico
m= 256 #tamaño de la imágen
n= 256 
G = np.fft.fft2(g)

p=np.array([[0 ,-1 ,0],[ -1, 4, -1],[0, -1, 0]])
Pp=np.fft.fft2(p,s=[m,n])

#valor de gamma se puede cambiar
#gamma=0.01 #la persona se ve menos nítida pero el fenómeno de gibbs es menor (el fenómeno de gibbs tmb genera ruido)
gamma=0.0005 #la persona se ve más nítida pero el fenómeno de gibbs es mayor
F=(np.conj(H)*G)/(abs(H)**2+gamma*abs(Pp)**2)
iRestored_P = np.real(np.fft.ifft2(F))

cv2.imshow('Imagen Restaurada con Filtro Parametrico', iRestored_P)
cv2.waitKey(0)






