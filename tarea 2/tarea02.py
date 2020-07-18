#--Instrucciones:--
#1. Se debe tener la imagen 'fallas.tif' en la misma carpeta donde se encuentra este archivo
#2. Se debe correr, y despues de cada seccion debería aparecer un(as) imágen(es)
#3. Se de sebe apretar cualquier tecla para pasar a los siguientes resultados.

import cv2
import matplotlib.pyplot as plt #para histograma
import numpy as np #libreria de manejo numerico, para funcion gama

# para el paso y 7: transformar la imagen binaria a valores entre 0 a 255.
def gamma_correction(img, factor):
    img = img/255.0
    img = cv2.pow(img, factor)
    return np.uint8(img*255)

#1 - Lectura de un imagen
img = cv2.imread('fallas.tif')

#transforma a niveles de grises
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Image Original', img_gray) 

cv2.waitKey()

#2 - Ecualización 
img_eq = cv2.equalizeHist(img_gray)

# ANALISIS --------- comente los resultados respecto a la imagen original
# Al ecualizar la imagen, se mejoran los contrastes de la imagen, es decir: 
# las áreas claras se tornan aún más claras y las áreas oscuras se tornan aún más oscuras en la 
# imágen ecualizada. Por lo tanto, queda más evidente lo que queremos observar en respecto al "fondo".
cv2.imshow('Imagen Ecualizada', img_eq) 

cv2.waitKey()

#3 - Filtro de la mediana
img_median = cv2.medianBlur(img_eq, 55)

cv2.imshow('Imagen Fitro Mediana - 55x',img_median) 

cv2.waitKey()

#4 - Reste la imagen del paso-3 con la del paso-2 y muestre el resultado. 
img_restada= cv2.subtract(img_median, img_eq)

# ANALISIS --------- analice en qué casos las fallas son más visibles.
# Los casos de las fallas más visibles son aquellas que en la imagen ecualizada eran más grandes o más oscuras.
# Esto, porque al restar la imagen "borrosa" de la ecualizada, nos quedamos solo con partes donde
# las diferencias son más notables. 
# Por ejemplo, en el centro de la imagen hay dos lineas horizontales las cuales solo son levemente diferenciable del fondo en la imagen original. 
# Entonces, al hacer el filtro de la mediana (generando la imagen "borrosa"), estos valores se mantuvieron similares y se restaron quedando en 0. 
cv2.imshow('Imagen Resta',img_restada) 

cv2.waitKey()

#5 - Binarizar
# Si el valor en escala de gris es superior a 30 [Negro. (= 0) y blanco (= 255)], será tomado como 1 (blanco). 
rt,img_bina = cv2.threshold(img_restada, 20,255,cv2.THRESH_BINARY)

cv2.imshow('Imagen Binarizada', img_bina) 

cv2.waitKey()

#6 - Laplaciano
img_laplace_restada = cv2.Laplacian(img_restada, cv2.CV_8U, ksize = 5)
img_laplace_bina = cv2.Laplacian(img_bina, cv2.CV_8U, ksize = 5)

# ANALISIS --------- analice en qué casos los bordes de las fallas se ven más claros y en qué casos no
# Cuando existe una mayor diferencia entre el fondo y el borde de las fallas, el laplaciano genera lineas más claras. 
# Por lo tanto, las burbujas más oscuras obtienen delimitaciones más claras que las burbujas más claras (las que se asemejan a los colores del fondo)
cv2.imshow('Imagen Laplaciano de Imagen Restada',img_laplace_restada)

cv2.waitKey()
# ANALISIS --------- analice en qué casos los bordes de las fallas se ven más claros y en qué casos no
# Como la imagen binarizada no genera rangos de tonalidades sino que solo 2 tonos: negro o blanco, 
# Los bordes de las fallas son todas de la misma tonalidad. El laplaciano genera un rango de valores para diferentes
# diferencias entre los bordes pero las diferencias son todas iguales en la imagen binarizada. 
cv2.imshow('Imagen Laplaciano de Imagen Binarizada',img_laplace_bina)

cv2.waitKey()

#7 - Multiplicación punto a punto img_bina con img_eq
#Recuerde transformar la imagen binaria a valores entre 0 a 255.
img_transformada = np.uint8(img_bina*255)

img_final = img_transformada * img_eq


cv2.imshow('Imagen Final',img_final)

cv2.waitKey()
cv2.destroyAllWindows()
