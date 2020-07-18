#--Instrucciones:--
#1. Se debe tener la imagen 'lagartija.jpg' en la misma carpeta donde se encuentra este archivo
#2. Se debe correr, y despues de cada seccion debería aparecer un(as) imágen(es)
#3. Se de sebe apretar la tecla 'q' para pasar a los siguientes resultados.


import cv2
import matplotlib.pyplot as plt #para histograma
import numpy as np #libreria de manejo numerico, para funcion gama

def gamma_correction(img, factor): #el programa comienza a funcionar dentro de la función
    img = img/255.0
    img = cv2.pow(img, factor)
    return np.uint8(img*255) #una vez que se alcanza return, se retorna el resultado a 'output'

#1 - Lectura de un imagen
img = cv2.imread('lagartija.jpg')
cv2.imshow('imagen RGB',img)

cv2.waitKey()
cv2.destroyAllWindows()

#2 - Tranformación a escala de grises
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('imagen gray',grayimg)

cv2.waitKey()
cv2.destroyAllWindows()

#3 - Separación de canales
blueimg, greenimg, redimg = cv2.split(img)

cv2.imshow('imagen RGB - canal R', redimg)
cv2.imshow('imagen RGB - canal G', greenimg)
cv2.imshow('imagen RGB - canal B', blueimg)

cv2.waitKey()
cv2.destroyAllWindows()

#4a - Histograma del canal R
hst = cv2.calcHist([img],[2],None,[256],[0,256])

plt.plot(hst) 
plt.show()

#4b - Ecualización del histograma del canal R
eqimg = cv2.equalizeHist(redimg)

cv2.imshow('Source image', redimg) 
cv2.imshow('Equalized Image', eqimg) 

cv2.waitKey()
cv2.destroyAllWindows()

#5 - Mejore la imagen del canal G obtenida en el paso 3 empleando la función gamma con valores 1.5 y 0.4. 
#Guarde estos resultados en distintos archivos.
funciongama1 = gamma_correction(greenimg, 1.5) #imagen mas oscura
funciongama2 = gamma_correction(greenimg, 0.4) #imagen mas clara

cv2.imshow('Gama 1.5',funciongama1)
cv2.imshow('Gama 0.4',funciongama1)

cv2.waitKey() 
cv2.destroyAllWindows()

#6 - Despliegue una imagen con el bit más significativo del paso 4 (utilice la imagen del paso 2) 
#Muestre por pantalla el resultado.

#para la representación de datos de 8 bits, hay 8 planos de bits: el primero contiene el conjunto de los bits más significativos, y el octavo contiene los bits menos significativos.
#k es el plano de bits al revez(desde 0 - 7: 7 es el bit más significativo)

#img.shape -> esta función nos permite crear una matriz del mismo tamaño de la imagen, con un valor específico de nivel de gris
                        #dimension(alto, ancho)  | valor deseado | tipo de datos
plane = np.full((eqimg.shape[0], eqimg.shape[1]), 2 ** 7, np.uint8)

#Multiplicamos cada nivel de gris por la imagen, o realizamos una operación binaria AND
res = plane & eqimg  

imgbitsignificativo = res*255
cv2.imshow("Plano bit #8 de imagen equalizada", imgbitsignificativo) 
cv2.imshow('Imagen equalizada', eqimg)

cv2.waitKey()
cv2.destroyAllWindows()
