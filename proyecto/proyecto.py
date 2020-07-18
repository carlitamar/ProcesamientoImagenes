# imagen es del siguiente tamaño 1856 * 1392 = 2.583.552

# OBJETIVO
# 1. Implementar un algoritmo de segmentacion de piel humana aplicado a un conjunto imagenes a color. 
# 2. Luego, evaluar el rendimiento de su implementación a través de la comparación entre imágenes 
# binarias con un conjunto de indicadores estadísticos.

import imutils
import numpy as np
import argparse
import cv2
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

#importar imagenes originales
img1 = cv2.imread('Caras01.jpg')
img2 = cv2.imread('Caras02.jpg')
img3 = cv2.imread('Caras03.jpg')
img4 = cv2.imread('Caras04.jpg')
img5 = cv2.imread('Caras05.jpg')
img6 = cv2.imread('Caras06.jpg')
img7 = cv2.imread('Caras07.jpg')

# Metodología 1
# Segmentación: Buscar en la literatura algoritmos de segmentación de piel humana variando distintos 
# parámetros como espacios de color, reducción del ruido, etc. 
# El resultado de la segmentación debe generar una máscara binaria que sirva de comparación con una 
# imagen que posee la segmentación ideal (provista para cada imagen de prueba).
# https://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/

# Segmentación según color de piel
t0 = time.time()

imgHSV = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
imgMascaraRango = cv2.inRange(imgHSV, lower, upper)

t1 = time.time()
total = t1-t0

cv2.imshow('Caras01: Segmentacion', imgMascaraRango)
print("Tiempo proceso segmentacion: ",total)
cv2.waitKey()

# Morfología
t0 = time.time()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
imgDilate = cv2.dilate(imgMascaraRango, kernel, iterations = 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
imgClausura = cv2.erode(imgDilate, kernel, iterations = 1)
imgClausura = cv2.GaussianBlur(imgClausura, (3, 3), 0)
imgMorfo = cv2.bitwise_and(img1, img1, mask = imgClausura)

t1 = time.time()
total = t1-t0

cv2.imshow('Caras01: Morfologia', imgMorfo)
print("Tiempo proceso morfologico: ",total)
cv2.waitKey()

# Binarización
t0 = time.time()

rt,imgBinarizada1 = cv2.threshold(imgMorfo,5,255,cv2.THRESH_BINARY) 

t1 = time.time()
total = t1-t0

cv2.imshow('Caras01: Binarizacion', imgBinarizada1)
print("Tiempo proceso binarización: ",total)
cv2.waitKey()

# Metodología 2
# Curvas del rendimiento: Calcular indicadores de rendimiento según la segmentación obtenida versus 
# la segmentación ideal. Para ello calcular los siguientes factores empleando las dos imágenes 
# binarias: Falsos Positivos, Falsos Negativos, Verdaderos Positivos, y Verdaderos Negativos. 
# Posteriormente determinar el índice F-Score para cada una de las siete imágenes del proyecto empleando 
# los factores antes descritos.
# https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python

# Caras01.jpg
# Segmentacion Ideal 
refer = cv2.imread('Refer01.bmp')

# -----------------------------
# FALSOS POSITIVOS (lo que detectamos como piel (lo blanco) erroneamente en rojo)
# resta entre imgBinarizada1 y refer
t0 = time.time()
diferencia_falsos_positivos = cv2.subtract(imgBinarizada1, refer)
# color the mask red
Conv_hsv_Gray = cv2.cvtColor(diferencia_falsos_positivos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_falsos_positivos[mask != 255] = [0, 0, 255]
t1 = time.time()
total = t1-t0
#mostrar imagen
cv2.imwrite('falsos_positivos1.png', diferencia_falsos_positivos)
cv2.imshow('Caras01: falsos positivos en rojo', diferencia_falsos_positivos)
cv2.waitKey()
# contar pixeles rojos
# https://stackoverflow.com/questions/42255410/how-to-count-the-number-of-pixels-with-a-certain-pixel-value-in-python-opencv
pixeles_rojos = np.count_nonzero((diferencia_falsos_positivos == [0, 0, 255]).all(axis = 2))
print("Falsos Positivos Caras01:",pixeles_rojos)
print("Tiempo segmentar falsos positivos piel: ",total)

# -----------------------------
#FALSOS NEGATIVOS (muestra en verde lo que detectamos como NO piel (lo negro) erroneamente)
# resta entre refer y imgBinarizada1
t0 = time.time()
diferencia_falsos_negativos = cv2.subtract(refer, imgBinarizada1)
# color the mask green
Conv_hsv_Gray = cv2.cvtColor(diferencia_falsos_negativos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_falsos_negativos[mask != 255] = [0, 255, 0]
t1 = time.time()
total = t1-t0
#mostrar imagen
cv2.imwrite('falsos_negativos1.png', diferencia_falsos_negativos)
cv2.imshow('Caras01: falsos negativos en verde', diferencia_falsos_negativos)
cv2.waitKey()
# contar pixeles verdes
pixeles_verdes = np.count_nonzero((diferencia_falsos_negativos == [0, 255, 0]).all(axis = 2))
print("Falsos Negativos Caras01:",pixeles_verdes)
print("Tiempo segmentar falsos negativos piel: ",total)

# -----------------------------
#VERDADEROS POSITIVOS (muestra en amarillo lo que detectamos como piel (lo blanco de la img binarizada) correctamente)
t0 = time.time()
falsonegativo = cv2.imread('falsos_negativos1.png')
# convert refer2 to green
refer_green = refer
Conv_hsv_Gray_refer = cv2.cvtColor(refer_green, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray_refer, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
refer_green[mask != 255] = [0, 255, 0]

# resta entre la segmentación ideal y la imagen de falsos negativos
diferencia_verdaderos_positivos = cv2.subtract(refer_green, falsonegativo)
# la mascara es amarilla
Conv_hsv_Gray = cv2.cvtColor(diferencia_verdaderos_positivos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_verdaderos_positivos[mask != 255] = [0, 255, 255]
t1 = time.time()
total = t1-t0
#mostrar imagen
cv2.imwrite('verdaderos_positivos1.png', diferencia_verdaderos_positivos)
cv2.imshow('Caras01: Verdaderos positivos en amarillo', diferencia_verdaderos_positivos)
cv2.waitKey()
# contar pixeles amarillos
pixeles_amarillos = np.count_nonzero((diferencia_verdaderos_positivos == [0, 255, 255]).all(axis = 2))
print("Verdaderos Postivos Caras01:",pixeles_amarillos)
print("Tiempo segmentar verdaderos positivos piel: ",total)

# -----------------------------
#VERDADEROS NEGATIVOS (lo que detectamos como NO piel (lo negro de la img binarizada) correctamente)
# suma entre la imagen binarizada y la imagen de segmentacion ideal
t0 = time.time()
suma_verdaderos_negativos = cv2.add(imgBinarizada1, refer)
# la mascara es azul
Conv_hsv_Gray = cv2.cvtColor(suma_verdaderos_negativos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
suma_verdaderos_negativos[mask != 0] = [255, 0, 0]
suma_verdaderos_negativos[mask != 255] = [0, 0, 0]
t1 = time.time()
total = t1-t0
#mostrar imagen
cv2.imwrite('verdaderos_negativos1.png', suma_verdaderos_negativos)
cv2.imshow('Caras01: Verdaderos negativos en negro', suma_verdaderos_negativos)
cv2.waitKey()
# contar pixeles azules
pixeles_azules = np.count_nonzero((suma_verdaderos_negativos == [255, 0, 0]).all(axis = 2))
print("Verdaderos Negativos Caras01:",pixeles_azules)
print("Tiempo segmentar verdaderos negativos piel: ",total)

# -----------------------------
# 2.583.552 pixeles totales
totalPixeles = 2583552

# determinar el índice F-Score 
# pixeles_rojos = false_p
# pixeles_verdes = false_n
# pixeles_amarillos = true_p
# pixeles_azules = true_n

#false negative rate
def calc_fnrate(false_n,true_p):
    fnrate = false_n/(false_n + true_p)
    
    return fnrate

#true positive rate or recall
def calc_recall(true_p,false_n):
    recall = true_p/(true_p + false_n)
    
    return recall

#true negative rate
def calc_tnrate(true_n,false_p):
    tnrate = true_n/(true_n + false_p)

    return tnrate

#false postive rate
def calc_fprate(false_p,true_n):
    fprate = false_p/(false_p + true_n)
    
    return fprate

def calc_precision(true_p,false_p):
    precision = true_p/(true_p + false_p)
    
    return precision

def calc_fscores(precision,recall):
    fscores = 2*(precision*recall)/(precision + recall)
    
    return fscores

fnrate = calc_fnrate(pixeles_verdes,pixeles_amarillos)
recall = calc_recall(pixeles_amarillos,pixeles_verdes)
tnrate = calc_tnrate(pixeles_azules,pixeles_rojos)
fprate = calc_fprate(pixeles_rojos,pixeles_azules)
precision = calc_precision(pixeles_amarillos,pixeles_rojos)
fscore_01= calc_fscores(precision,recall)

print("El false negative rate para Caras01 es: ",fnrate)
print("El true postive rate o recall Caras01 es: ",recall)
print("El true negative rate para Caras01 es: ",tnrate)
print("El false positive rate para Caras01 es: ",fprate)
print("El precision para Caras01 es: ",precision)
print("El fscore para Caras01 es: ",fscore_01)
# ----------------------------------------------------------------------------------------------------------------------------------------
# Cara02.jpg
# Metodología 1
# Segmentación según color de piel
imgHSV = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
imgMascaraRango = cv2.inRange(imgHSV, lower, upper)

cv2.imshow('Caras02: Segmentacion', imgMascaraRango)
cv2.waitKey()

# Morfología
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
imgDilate = cv2.dilate(imgMascaraRango, kernel, iterations = 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
imgClausura = cv2.erode(imgDilate, kernel, iterations = 1)
imgClausura = cv2.GaussianBlur(imgClausura, (3, 3), 0)
imgMorfo = cv2.bitwise_and(img2, img2, mask = imgClausura)

cv2.imshow('Caras02: Morfologia', imgMorfo)
cv2.waitKey()

# Binarización
rt,imgBinarizada2 = cv2.threshold(imgMorfo,5,255,cv2.THRESH_BINARY) 

cv2.imshow('Caras02: Binarizacion', imgBinarizada2)
cv2.waitKey()

# Metodología 2
# Segmentacion Ideal 
refer = cv2.imread('Refer02.bmp')

# -----------------------------
# FALSOS POSITIVOS (lo que detectamos como piel (lo blanco) erroneamente en rojo)
# resta entre imgBinarizada2 y refer
diferencia_falsos_positivos = cv2.subtract(imgBinarizada2, refer)
# color the mask red
Conv_hsv_Gray = cv2.cvtColor(diferencia_falsos_positivos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_falsos_positivos[mask != 255] = [0, 0, 255]
#mostrar imagen
cv2.imwrite('falsos_positivos2.png', diferencia_falsos_positivos)
cv2.imshow('Caras02: falsos positivos en rojo', diferencia_falsos_positivos)
cv2.waitKey()
# contar pixeles rojos
# https://stackoverflow.com/questions/42255410/how-to-count-the-number-of-pixels-with-a-certain-pixel-value-in-python-opencv
pixeles_rojos = np.count_nonzero((diferencia_falsos_positivos == [0, 0, 255]).all(axis = 2))
print("Falsos Positivos Caras02:",pixeles_rojos)

# -----------------------------
# FALSOS NEGATIVOS (muestra en verde lo que detectamos como NO piel (lo negro) erroneamente)
# resta entre refer y imgBinarizada2 
diferencia_falsos_negativos = cv2.subtract(refer, imgBinarizada2)
# color the mask green
Conv_hsv_Gray = cv2.cvtColor(diferencia_falsos_negativos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_falsos_negativos[mask != 255] = [0, 255, 0]
#mostrar imagen
cv2.imwrite('falsos_negativos2.png', diferencia_falsos_negativos)
cv2.imshow('Caras02: falsos negativos en verde', diferencia_falsos_negativos)
cv2.waitKey()
# contar pixeles verdes
pixeles_verdes = np.count_nonzero((diferencia_falsos_negativos == [0, 255, 0]).all(axis = 2))
print("Falsos Negativos Caras02:",pixeles_verdes)

# -----------------------------
#VERDADEROS POSITIVOS (muestra en amarillo lo que detectamos como piel (lo blanco de la img binarizada) correctamente)
falsonegativo = cv2.imread('falsos_negativos2.png')
# convert refer2 to green
refer_green = refer
Conv_hsv_Gray_refer = cv2.cvtColor(refer_green, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray_refer, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
refer_green[mask != 255] = [0, 255, 0]

# resta entre la segmentación ideal y la imagen de falsos negativos
diferencia_verdaderos_positivos = cv2.subtract(refer_green, falsonegativo)
# la mascara es amarilla
Conv_hsv_Gray = cv2.cvtColor(diferencia_verdaderos_positivos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_verdaderos_positivos[mask != 255] = [0, 255, 255]
#mostrar imagen
cv2.imwrite('verdaderos_positivos2.png', diferencia_verdaderos_positivos)
cv2.imshow('Caras02: Verdaderos positivos en amarillo', diferencia_verdaderos_positivos)
cv2.waitKey()
# contar pixeles amarillos
pixeles_amarillos = np.count_nonzero((diferencia_verdaderos_positivos == [0, 255, 255]).all(axis = 2))
print("Verdaderos Postivos Caras02:",pixeles_amarillos)
# -----------------------------
#VERDADEROS NEGATIVOS (lo que detectamos como NO piel (lo negro de la img binarizada) correctamente)

# suma entre la imagen binarizada y la imagen de segmentacion ideal
suma_verdaderos_negativos = cv2.add(imgBinarizada2, refer)
# la mascara es azul
Conv_hsv_Gray = cv2.cvtColor(suma_verdaderos_negativos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
suma_verdaderos_negativos[mask != 0] = [255, 0, 0]
suma_verdaderos_negativos[mask != 255] = [0, 0, 0]
#mostrar imagen
cv2.imwrite('verdaderos_negativos2.png', suma_verdaderos_negativos)
cv2.imshow('Caras02: Verdaderos negativos en negro', suma_verdaderos_negativos)
cv2.waitKey()
# contar pixeles azules
pixeles_azules = np.count_nonzero((suma_verdaderos_negativos == [255, 0, 0]).all(axis = 2))
print("Verdaderos Negativos Caras02:",pixeles_azules)

#valores
fnrate = calc_fnrate(pixeles_verdes,pixeles_amarillos)
recall = calc_recall(pixeles_amarillos,pixeles_verdes)
tnrate = calc_tnrate(pixeles_azules,pixeles_rojos)
fprate = calc_fprate(pixeles_rojos,pixeles_azules)
precision = calc_precision(pixeles_amarillos,pixeles_rojos)

t0 = time.time()
fscore_02= calc_fscores(precision,recall)
t1 = time.time()
total = t1-t0
print("El false negative rate para Caras02 es: ",fnrate)
print("El true postive rate o recall Caras02 es: ",recall)
print("El true negative rate para Caras02 es: ",tnrate)
print("El false positive rate para Caras02 es: ",fprate)
print("El precision para Caras02 es: ",precision)
print("El fscore para Caras02 es: ",fscore_02)
# ----------------------------------------------------------------------------------------------------------------------------------------
# Caras03.jpg
# Metodología 1
# Segmentación según color de piel
imgHSV = cv2.cvtColor(img3,cv2.COLOR_BGR2HSV)
imgMascaraRango = cv2.inRange(imgHSV, lower, upper)

cv2.imshow('Caras03: Segmentacion', imgMascaraRango)
cv2.waitKey()

# Morfología
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
imgDilate = cv2.dilate(imgMascaraRango, kernel, iterations = 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
imgClausura = cv2.erode(imgDilate, kernel, iterations = 1)
imgClausura = cv2.GaussianBlur(imgClausura, (3, 3), 0)
imgMorfo = cv2.bitwise_and(img3, img3, mask = imgClausura)

cv2.imshow('Caras03: Morfologia', imgMorfo)
cv2.waitKey()

# Binarización
rt,imgBinarizada3 = cv2.threshold(imgMorfo,5,255,cv2.THRESH_BINARY) 

cv2.imshow('Caras03: Binarizacion', imgBinarizada3)
cv2.waitKey()

# Metodología 2
# Segmentacion Ideal 
refer = cv2.imread('Refer03.bmp')

# -----------------------------
# FALSOS POSITIVOS (lo que detectamos como piel (lo blanco) erroneamente en rojo)
# resta entre imgBinarizada3 y refer
diferencia_falsos_positivos = cv2.subtract(imgBinarizada3, refer)
# color the mask red
Conv_hsv_Gray = cv2.cvtColor(diferencia_falsos_positivos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_falsos_positivos[mask != 255] = [0, 0, 255]
#mostrar imagen
cv2.imwrite('falsos_positivos3.png', diferencia_falsos_positivos)
cv2.imshow('Caras03: falsos positivos en rojo', diferencia_falsos_positivos)
cv2.waitKey()
# contar pixeles rojos
# https://stackoverflow.com/questions/42255410/how-to-count-the-number-of-pixels-with-a-certain-pixel-value-in-python-opencv
pixeles_rojos = np.count_nonzero((diferencia_falsos_positivos == [0, 0, 255]).all(axis = 2))
print("Falsos Positivos Caras03:",pixeles_rojos)

# -----------------------------
# FALSOS NEGATIVOS (muestra en verde lo que detectamos como NO piel (lo negro) erroneamente)
# resta entre refer y imgBinarizada3 
diferencia_falsos_negativos = cv2.subtract(refer, imgBinarizada3)
# color the mask green
Conv_hsv_Gray = cv2.cvtColor(diferencia_falsos_negativos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_falsos_negativos[mask != 255] = [0, 255, 0]
#mostrar imagen
cv2.imwrite('falsos_negativos3.png', diferencia_falsos_negativos)
cv2.imshow('Caras03: falsos negativos en verde', diferencia_falsos_negativos)
cv2.waitKey()
# contar pixeles verdes
pixeles_verdes = np.count_nonzero((diferencia_falsos_negativos == [0, 255, 0]).all(axis = 2))
print("Falsos Negativos Caras03:",pixeles_verdes)

# -----------------------------
#VERDADEROS POSITIVOS (muestra en amarillo lo que detectamos como piel (lo blanco de la img binarizada) correctamente)
falsonegativo = cv2.imread('falsos_negativos3.png')
# convert refer2 to green
refer_green = refer
Conv_hsv_Gray_refer = cv2.cvtColor(refer_green, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray_refer, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
refer_green[mask != 255] = [0, 255, 0]

# resta entre la segmentación ideal y la imagen de falsos negativos
diferencia_verdaderos_positivos = cv2.subtract(refer_green, falsonegativo)
# la mascara es amarilla
Conv_hsv_Gray = cv2.cvtColor(diferencia_verdaderos_positivos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_verdaderos_positivos[mask != 255] = [0, 255, 255]
#mostrar imagen
cv2.imwrite('verdaderos_positivos3.png', diferencia_verdaderos_positivos)
cv2.imshow('Caras03: Verdaderos positivos en amarillo', diferencia_verdaderos_positivos)
cv2.waitKey()
# contar pixeles amarillos
pixeles_amarillos = np.count_nonzero((diferencia_verdaderos_positivos == [0, 255, 255]).all(axis = 2))
print("Verdaderos Postivos Caras03:",pixeles_amarillos)

# -----------------------------
#VERDADEROS NEGATIVOS (lo que detectamos como NO piel (lo negro de la img binarizada) correctamente)

# suma entre la imagen binarizada y la imagen de segmentacion ideal
suma_verdaderos_negativos = cv2.add(imgBinarizada3, refer)
# la mascara es azul
Conv_hsv_Gray = cv2.cvtColor(suma_verdaderos_negativos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
suma_verdaderos_negativos[mask != 0] = [255, 0, 0]
suma_verdaderos_negativos[mask != 255] = [0, 0, 0]
#mostrar imagen
cv2.imwrite('verdaderos_negativos3.png', suma_verdaderos_negativos)
cv2.imshow('Caras03: Verdaderos negativos en negro', suma_verdaderos_negativos)
cv2.waitKey()
# contar pixeles azules
pixeles_azules = np.count_nonzero((suma_verdaderos_negativos == [255, 0, 0]).all(axis = 2))
print("Verdaderos Negativos Caras03:",pixeles_azules)

#valores
fnrate = calc_fnrate(pixeles_verdes,pixeles_amarillos)
recall = calc_recall(pixeles_amarillos,pixeles_verdes)
tnrate = calc_tnrate(pixeles_azules,pixeles_rojos)
fprate = calc_fprate(pixeles_rojos,pixeles_azules)
precision = calc_precision(pixeles_amarillos,pixeles_rojos)

t0 = time.time()
fscore_03= calc_fscores(precision,recall)
t1 = time.time()
total = t1-t0
print("El false negative rate para Caras03 es: ",fnrate)
print("El true postive rate o recall Caras03 es: ",recall)
print("El true negative rate para Caras03 es: ",tnrate)
print("El false positive rate para Caras03 es: ",fprate)
print("El precision para Caras03 es: ",precision)
print("El fscore para Caras03 es: ",fscore_03)
# ----------------------------------------------------------------------------------------------------------------------------------------
# Caras04.jpg
# Metodología 1
# Segmentación según color de piel
imgHSV = cv2.cvtColor(img4,cv2.COLOR_BGR2HSV)
imgMascaraRango = cv2.inRange(imgHSV, lower, upper)

cv2.imshow('Caras04: Segmentacion', imgMascaraRango)
cv2.waitKey()

# Morfología
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
imgDilate = cv2.dilate(imgMascaraRango, kernel, iterations = 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
imgClausura = cv2.erode(imgDilate, kernel, iterations = 1)
imgClausura = cv2.GaussianBlur(imgClausura, (3, 3), 0)
imgMorfo = cv2.bitwise_and(img4, img4, mask = imgClausura)

cv2.imshow('Caras04: Morfologia', imgMorfo)
cv2.waitKey()

# Binarización
rt,imgBinarizada4 = cv2.threshold(imgMorfo,5,255,cv2.THRESH_BINARY) 

cv2.imshow('Caras04: Binarizacion', imgBinarizada4)
cv2.waitKey()

# Metodología 2
# Segmentacion Ideal 
refer = cv2.imread('Refer04.bmp')

# -----------------------------
# FALSOS POSITIVOS (lo que detectamos como piel (lo blanco) erroneamente en rojo)
# resta entre imgBinarizada4 y refer
diferencia_falsos_positivos = cv2.subtract(imgBinarizada4, refer)
# color the mask red
Conv_hsv_Gray = cv2.cvtColor(diferencia_falsos_positivos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_falsos_positivos[mask != 255] = [0, 0, 255]
#mostrar imagen
cv2.imwrite('falsos_positivos4.png', diferencia_falsos_positivos)
cv2.imshow('Caras04: falsos positivos en rojo', diferencia_falsos_positivos)
cv2.waitKey()
# contar pixeles rojos
# https://stackoverflow.com/questions/42255410/how-to-count-the-number-of-pixels-with-a-certain-pixel-value-in-python-opencv
pixeles_rojos = np.count_nonzero((diferencia_falsos_positivos == [0, 0, 255]).all(axis = 2))
print("Falsos Positivos Caras04:",pixeles_rojos)

# -----------------------------
# FALSOS NEGATIVOS (muestra en verde lo que detectamos como NO piel (lo negro) erroneamente)
# resta entre refer y imgBinarizada4 
diferencia_falsos_negativos = cv2.subtract(refer, imgBinarizada4)
# color the mask green
Conv_hsv_Gray = cv2.cvtColor(diferencia_falsos_negativos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_falsos_negativos[mask != 255] = [0, 255, 0]
#mostrar imagen
cv2.imwrite('falsos_negativos4.png', diferencia_falsos_negativos)
cv2.imshow('Caras04: falsos negativos en verde', diferencia_falsos_negativos)
cv2.waitKey()
# contar pixeles verdes
pixeles_verdes = np.count_nonzero((diferencia_falsos_negativos == [0, 255, 0]).all(axis = 2))
print("Falsos Negativos Caras04:",pixeles_verdes)

# -----------------------------
#VERDADEROS POSITIVOS (muestra en amarillo lo que detectamos como piel (lo blanco de la img binarizada) correctamente)
falsonegativo = cv2.imread('falsos_negativos4.png')
# convert refer2 to green
refer_green = refer
Conv_hsv_Gray_refer = cv2.cvtColor(refer_green, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray_refer, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
refer_green[mask != 255] = [0, 255, 0]

# resta entre la segmentación ideal y la imagen de falsos negativos
diferencia_verdaderos_positivos = cv2.subtract(refer_green, falsonegativo)
# la mascara es amarilla
Conv_hsv_Gray = cv2.cvtColor(diferencia_verdaderos_positivos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_verdaderos_positivos[mask != 255] = [0, 255, 255]
#mostrar imagen
cv2.imwrite('verdaderos_positivos4.png', diferencia_verdaderos_positivos)
cv2.imshow('Caras04: Verdaderos positivos en amarillo', diferencia_verdaderos_positivos)
cv2.waitKey()
# contar pixeles amarillos
pixeles_amarillos = np.count_nonzero((diferencia_verdaderos_positivos == [0, 255, 255]).all(axis = 2))
print("Verdaderos Postivos Caras04:",pixeles_amarillos)

# -----------------------------
#VERDADEROS NEGATIVOS (lo que detectamos como NO piel (lo negro de la img binarizada) correctamente)

# suma entre la imagen binarizada y la imagen de segmentacion ideal
suma_verdaderos_negativos = cv2.add(imgBinarizada4, refer)
# la mascara es azul
Conv_hsv_Gray = cv2.cvtColor(suma_verdaderos_negativos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
suma_verdaderos_negativos[mask != 0] = [255, 0, 0]
suma_verdaderos_negativos[mask != 255] = [0, 0, 0]
#mostrar imagen
cv2.imwrite('verdaderos_negativos4.png', suma_verdaderos_negativos)
cv2.imshow('Caras04: Verdaderos negativos en negro', suma_verdaderos_negativos)
cv2.waitKey()
# contar pixeles azules
pixeles_azules = np.count_nonzero((suma_verdaderos_negativos == [255, 0, 0]).all(axis = 2))
print("Verdaderos Negativos Caras04:",pixeles_azules)

#valores
fnrate = calc_fnrate(pixeles_verdes,pixeles_amarillos)
recall = calc_recall(pixeles_amarillos,pixeles_verdes)
tnrate = calc_tnrate(pixeles_azules,pixeles_rojos)
fprate = calc_fprate(pixeles_rojos,pixeles_azules)
precision = calc_precision(pixeles_amarillos,pixeles_rojos)

t0 = time.time()
fscore_04= calc_fscores(precision,recall)
t1 = time.time()
total = t1-t0
print("El false negative rate para Caras04 es: ",fnrate)
print("El true postive rate o recall Caras04 es: ",recall)
print("El true negative rate para Caras04 es: ",tnrate)
print("El false positive rate para Caras04 es: ",fprate)
print("El precision para Caras04 es: ",precision)
print("El fscore para Caras04 es: ",fscore_04)
# ----------------------------------------------------------------------------------------------------------------------------------------
# Caras05.jpg
# Metodología 1
# Segmentación según color de piel
imgHSV = cv2.cvtColor(img5,cv2.COLOR_BGR2HSV)
imgMascaraRango = cv2.inRange(imgHSV, lower, upper)

cv2.imshow('Caras05: Segmentacion', imgMascaraRango)
cv2.waitKey()

# Morfología
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
imgDilate = cv2.dilate(imgMascaraRango, kernel, iterations = 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
imgClausura = cv2.erode(imgDilate, kernel, iterations = 1)
imgClausura = cv2.GaussianBlur(imgClausura, (3, 3), 0)
imgMorfo = cv2.bitwise_and(img5, img5, mask = imgClausura)

cv2.imshow('Caras05: Morfologia', imgMorfo)
cv2.waitKey()

# Binarización
rt,imgBinarizada5 = cv2.threshold(imgMorfo,5,255,cv2.THRESH_BINARY) 

cv2.imshow('Caras05: Binarizacion', imgBinarizada5)
cv2.waitKey()

# Metodología 2
# Segmentacion Ideal 
refer = cv2.imread('Refer05.bmp')

# -----------------------------
# FALSOS POSITIVOS (lo que detectamos como piel (lo blanco) erroneamente en rojo)
# resta entre imgBinarizada5 y refer
diferencia_falsos_positivos = cv2.subtract(imgBinarizada5, refer)
# color the mask red
Conv_hsv_Gray = cv2.cvtColor(diferencia_falsos_positivos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_falsos_positivos[mask != 255] = [0, 0, 255]
#mostrar imagen
cv2.imwrite('falsos_positivos5.png', diferencia_falsos_positivos)
cv2.imshow('Caras05: falsos positivos en rojo', diferencia_falsos_positivos)
cv2.waitKey()
# contar pixeles rojos
# https://stackoverflow.com/questions/42255410/how-to-count-the-number-of-pixels-with-a-certain-pixel-value-in-python-opencv
pixeles_rojos = np.count_nonzero((diferencia_falsos_positivos == [0, 0, 255]).all(axis = 2))
print("Falsos Positivos Caras05:",pixeles_rojos)

# -----------------------------
# FALSOS NEGATIVOS (muestra en verde lo que detectamos como NO piel (lo negro) erroneamente)
# resta entre refer y imgBinarizada5 
diferencia_falsos_negativos = cv2.subtract(refer, imgBinarizada5)
# color the mask green
Conv_hsv_Gray = cv2.cvtColor(diferencia_falsos_negativos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_falsos_negativos[mask != 255] = [0, 255, 0]
#mostrar imagen
cv2.imwrite('falsos_negativos5.png', diferencia_falsos_negativos)
cv2.imshow('Caras05: falsos negativos en verde', diferencia_falsos_negativos)
cv2.waitKey()
# contar pixeles verdes
pixeles_verdes = np.count_nonzero((diferencia_falsos_negativos == [0, 255, 0]).all(axis = 2))
print("Falsos Negativos Caras05:",pixeles_verdes)

# -----------------------------
#VERDADEROS POSITIVOS (muestra en amarillo lo que detectamos como piel (lo blanco de la img binarizada) correctamente)
falsonegativo = cv2.imread('falsos_negativos5.png')
# convert refer2 to green
refer_green = refer
Conv_hsv_Gray_refer = cv2.cvtColor(refer_green, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray_refer, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
refer_green[mask != 255] = [0, 255, 0]

# resta entre la segmentación ideal y la imagen de falsos negativos
diferencia_verdaderos_positivos = cv2.subtract(refer_green, falsonegativo)
# la mascara es amarilla
Conv_hsv_Gray = cv2.cvtColor(diferencia_verdaderos_positivos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_verdaderos_positivos[mask != 255] = [0, 255, 255]
#mostrar imagen
cv2.imwrite('verdaderos_positivos5.png', diferencia_verdaderos_positivos)
cv2.imshow('Caras05: Verdaderos positivos en amarillo', diferencia_verdaderos_positivos)
cv2.waitKey()
# contar pixeles amarillos
pixeles_amarillos = np.count_nonzero((diferencia_verdaderos_positivos == [0, 255, 255]).all(axis = 2))
print("Verdaderos Postivos Caras05:",pixeles_amarillos)

# -----------------------------
#VERDADEROS NEGATIVOS (lo que detectamos como NO piel (lo negro de la img binarizada) correctamente)

# suma entre la imagen binarizada y la imagen de segmentacion ideal
suma_verdaderos_negativos = cv2.add(imgBinarizada5, refer)
# la mascara es azul
Conv_hsv_Gray = cv2.cvtColor(suma_verdaderos_negativos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
suma_verdaderos_negativos[mask != 0] = [255, 0, 0]
suma_verdaderos_negativos[mask != 255] = [0, 0, 0]
#mostrar imagen
cv2.imwrite('verdaderos_negativos5.png', suma_verdaderos_negativos)
cv2.imshow('Caras05: Verdaderos negativos en negro', suma_verdaderos_negativos)
cv2.waitKey()
# contar pixeles azules
pixeles_azules = np.count_nonzero((suma_verdaderos_negativos == [255, 0, 0]).all(axis = 2))
print("Verdaderos Negativos Caras05:",pixeles_azules)

# -----------------------------
#valores
fnrate = calc_fnrate(pixeles_verdes,pixeles_amarillos)
recall = calc_recall(pixeles_amarillos,pixeles_verdes)
tnrate = calc_tnrate(pixeles_azules,pixeles_rojos)
fprate = calc_fprate(pixeles_rojos,pixeles_azules)
precision = calc_precision(pixeles_amarillos,pixeles_rojos)

t0 = time.time()
fscore_05= calc_fscores(precision,recall)
t1 = time.time()
total = t1-t0
print("El false negative rate para Caras05 es: ",fnrate)
print("El true postive rate o recall Caras05 es: ",recall)
print("El true negative rate para Caras05 es: ",tnrate)
print("El false positive rate para Caras05 es: ",fprate)
print("El precision para Caras05 es: ",precision)
print("El fscore para Caras05 es: ",fscore_05)
# ----------------------------------------------------------------------------------------------------------------------------------------
# Caras06.jpg
# Metodología 1
# Segmentación según color de piel
imgHSV = cv2.cvtColor(img6,cv2.COLOR_BGR2HSV)
imgMascaraRango = cv2.inRange(imgHSV, lower, upper)

cv2.imshow('Caras06: Segmentacion', imgMascaraRango)
cv2.waitKey()

# Morfología
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
imgDilate = cv2.dilate(imgMascaraRango, kernel, iterations = 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
imgClausura = cv2.erode(imgDilate, kernel, iterations = 1)
imgClausura = cv2.GaussianBlur(imgClausura, (3, 3), 0)
imgMorfo = cv2.bitwise_and(img6, img6, mask = imgClausura)

cv2.imshow('Caras06: Morfologia', imgMorfo)
cv2.waitKey()

# Binarización
rt,imgBinarizada6 = cv2.threshold(imgMorfo,5,255,cv2.THRESH_BINARY) 

cv2.imshow('Caras06: Binarizacion', imgBinarizada6)
cv2.waitKey()

# Metodología 2
# Segmentacion Ideal 
refer = cv2.imread('Refer06.bmp')

# -----------------------------
# FALSOS POSITIVOS (lo que detectamos como piel (lo blanco) erroneamente en rojo)
# resta entre imgBinarizada6 y refer
diferencia_falsos_positivos = cv2.subtract(imgBinarizada6, refer)
# color the mask red
Conv_hsv_Gray = cv2.cvtColor(diferencia_falsos_positivos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_falsos_positivos[mask != 255] = [0, 0, 255]
#mostrar imagen
cv2.imwrite('falsos_positivos6.png', diferencia_falsos_positivos)
cv2.imshow('Caras06: falsos positivos en rojo', diferencia_falsos_positivos)
cv2.waitKey()
# contar pixeles rojos
# https://stackoverflow.com/questions/42255410/how-to-count-the-number-of-pixels-with-a-certain-pixel-value-in-python-opencv
pixeles_rojos = np.count_nonzero((diferencia_falsos_positivos == [0, 0, 255]).all(axis = 2))
print("Falsos Positivos Caras06:",pixeles_rojos)

# -----------------------------
# FALSOS NEGATIVOS (muestra en verde lo que detectamos como NO piel (lo negro) erroneamente)
# resta entre refer y imgBinarizada6 
diferencia_falsos_negativos = cv2.subtract(refer, imgBinarizada6)
# color the mask green
Conv_hsv_Gray = cv2.cvtColor(diferencia_falsos_negativos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_falsos_negativos[mask != 255] = [0, 255, 0]
#mostrar imagen
cv2.imwrite('falsos_negativos6.png', diferencia_falsos_negativos)
cv2.imshow('Caras06: falsos negativos en verde', diferencia_falsos_negativos)
cv2.waitKey()
# contar pixeles verdes
pixeles_verdes = np.count_nonzero((diferencia_falsos_negativos == [0, 255, 0]).all(axis = 2))
print("Falsos Negativos Caras06:",pixeles_verdes)

# -----------------------------
#VERDADEROS POSITIVOS (muestra en amarillo lo que detectamos como piel (lo blanco de la img binarizada) correctamente)
falsonegativo = cv2.imread('falsos_negativos6.png')
# convert refer2 to green
refer_green = refer
Conv_hsv_Gray_refer = cv2.cvtColor(refer_green, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray_refer, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
refer_green[mask != 255] = [0, 255, 0]

# resta entre la segmentación ideal y la imagen de falsos negativos
diferencia_verdaderos_positivos = cv2.subtract(refer_green, falsonegativo)
# la mascara es amarilla
Conv_hsv_Gray = cv2.cvtColor(diferencia_verdaderos_positivos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_verdaderos_positivos[mask != 255] = [0, 255, 255]
#mostrar imagen
cv2.imwrite('verdaderos_positivos6.png', diferencia_verdaderos_positivos)
cv2.imshow('Caras06: Verdaderos positivos en amarillo', diferencia_verdaderos_positivos)
cv2.waitKey()
# contar pixeles amarillos
pixeles_amarillos = np.count_nonzero((diferencia_verdaderos_positivos == [0, 255, 255]).all(axis = 2))
print("Verdaderos Postivos Caras06:",pixeles_amarillos)

# -----------------------------
#VERDADEROS NEGATIVOS (lo que detectamos como NO piel (lo negro de la img binarizada) correctamente)

# suma entre la imagen binarizada y la imagen de segmentacion ideal
suma_verdaderos_negativos = cv2.add(imgBinarizada6, refer)
# la mascara es azul
Conv_hsv_Gray = cv2.cvtColor(suma_verdaderos_negativos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
suma_verdaderos_negativos[mask != 0] = [255, 0, 0]
suma_verdaderos_negativos[mask != 255] = [0, 0, 0]
#mostrar imagen
cv2.imwrite('verdaderos_negativos6.png', suma_verdaderos_negativos)
cv2.imshow('Caras06: Verdaderos negativos en negro', suma_verdaderos_negativos)
cv2.waitKey()
# contar pixeles azules
pixeles_azules = np.count_nonzero((suma_verdaderos_negativos == [255, 0, 0]).all(axis = 2))
print("Verdaderos Negativos Caras06:",pixeles_azules)

#valores
fnrate = calc_fnrate(pixeles_verdes,pixeles_amarillos)
recall = calc_recall(pixeles_amarillos,pixeles_verdes)
tnrate = calc_tnrate(pixeles_azules,pixeles_rojos)
fprate = calc_fprate(pixeles_rojos,pixeles_azules)
precision = calc_precision(pixeles_amarillos,pixeles_rojos)

t0 = time.time()
fscore_06= calc_fscores(precision,recall)
t1 = time.time()
total = t1-t0
print("El false negative rate para Caras06 es: ",fnrate)
print("El true postive rate o recall Caras06 es: ",recall)
print("El true negative rate para Caras06 es: ",tnrate)
print("El false positive rate para Caras06 es: ",fprate)
print("El precision para Caras06 es: ",precision)
print("El fscore para Caras06 es: ",fscore_06)
# ----------------------------------------------------------------------------------------------------------------------------------------
# Caras07.jpg
# Metodología 1
# Segmentación según color de piel
imgHSV = cv2.cvtColor(img7,cv2.COLOR_BGR2HSV)
imgMascaraRango = cv2.inRange(imgHSV, lower, upper)

cv2.imshow('Caras07: Segmentacion', imgMascaraRango)
cv2.waitKey()

# Morfología
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
imgDilate = cv2.dilate(imgMascaraRango, kernel, iterations = 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
imgClausura = cv2.erode(imgDilate, kernel, iterations = 1)
imgClausura = cv2.GaussianBlur(imgClausura, (3, 3), 0)
imgMorfo = cv2.bitwise_and(img7, img7, mask = imgClausura)

cv2.imshow('Caras07: Morfologia', imgMorfo)
cv2.waitKey()

# Binarización
rt,imgBinarizada7 = cv2.threshold(imgMorfo,5,255,cv2.THRESH_BINARY) 

cv2.imshow('Caras07: Binarizacion', imgBinarizada7)
cv2.waitKey()

# Metodología 2
# Segmentacion Ideal 
refer = cv2.imread('Refer07.bmp')

# -----------------------------
# FALSOS POSITIVOS (lo que detectamos como piel (lo blanco) erroneamente en rojo)
# resta entre imgBinarizada7 y refer
diferencia_falsos_positivos = cv2.subtract(imgBinarizada7, refer)
# color the mask red
Conv_hsv_Gray = cv2.cvtColor(diferencia_falsos_positivos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_falsos_positivos[mask != 255] = [0, 0, 255]
#mostrar imagen
cv2.imwrite('falsos_positivos7.png', diferencia_falsos_positivos)
cv2.imshow('Caras07: falsos positivos en rojo', diferencia_falsos_positivos)
cv2.waitKey()
# contar pixeles rojos
# https://stackoverflow.com/questions/42255410/how-to-count-the-number-of-pixels-with-a-certain-pixel-value-in-python-opencv
pixeles_rojos = np.count_nonzero((diferencia_falsos_positivos == [0, 0, 255]).all(axis = 2))
print("Falsos Positivos Caras07:",pixeles_rojos)

# -----------------------------
# FALSOS NEGATIVOS (muestra en verde lo que detectamos como NO piel (lo negro) erroneamente)
# resta entre refer y imgBinarizada7 
diferencia_falsos_negativos = cv2.subtract(refer, imgBinarizada7)
# color the mask green
Conv_hsv_Gray = cv2.cvtColor(diferencia_falsos_negativos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_falsos_negativos[mask != 255] = [0, 255, 0]
#mostrar imagen
cv2.imwrite('falsos_negativos7.png', diferencia_falsos_negativos)
cv2.imshow('Caras07: falsos negativos en verde', diferencia_falsos_negativos)
cv2.waitKey()
# contar pixeles verdes
pixeles_verdes = np.count_nonzero((diferencia_falsos_negativos == [0, 255, 0]).all(axis = 2))
print("Falsos Negativos Caras07:",pixeles_verdes)

# -----------------------------
#VERDADEROS POSITIVOS (muestra en amarillo lo que detectamos como piel (lo blanco de la img binarizada) correctamente)
falsonegativo = cv2.imread('falsos_negativos7.png')
# convert refer2 to green
refer_green = refer
Conv_hsv_Gray_refer = cv2.cvtColor(refer_green, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray_refer, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
refer_green[mask != 255] = [0, 255, 0]

# resta entre la segmentación ideal y la imagen de falsos negativos
diferencia_verdaderos_positivos = cv2.subtract(refer_green, falsonegativo)
# la mascara es amarilla
Conv_hsv_Gray = cv2.cvtColor(diferencia_verdaderos_positivos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
diferencia_verdaderos_positivos[mask != 255] = [0, 255, 255]
#mostrar imagen
cv2.imwrite('verdaderos_positivos7.png', diferencia_verdaderos_positivos)
cv2.imshow('Caras07: Verdaderos positivos en amarillo', diferencia_verdaderos_positivos)
cv2.waitKey()
# contar pixeles amarillos
pixeles_amarillos = np.count_nonzero((diferencia_verdaderos_positivos == [0, 255, 255]).all(axis = 2))
print("Verdaderos Postivos Caras07:",pixeles_amarillos)

# -----------------------------
#VERDADEROS NEGATIVOS (lo que detectamos como NO piel (lo negro de la img binarizada) correctamente)

# suma entre la imagen binarizada y la imagen de segmentacion ideal
suma_verdaderos_negativos = cv2.add(imgBinarizada7, refer)
# la mascara es azul
Conv_hsv_Gray = cv2.cvtColor(suma_verdaderos_negativos, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
suma_verdaderos_negativos[mask != 0] = [255, 0, 0]
suma_verdaderos_negativos[mask != 255] = [0, 0, 0]
#mostrar imagen
cv2.imwrite('verdaderos_negativos7.png', suma_verdaderos_negativos)
cv2.imshow('Caras07: Verdaderos negativos en negro', suma_verdaderos_negativos)
cv2.waitKey()
# contar pixeles azules
pixeles_azules = np.count_nonzero((suma_verdaderos_negativos == [255, 0, 0]).all(axis = 2))
print("Verdaderos Negativos Caras07:",pixeles_azules)

#valores
fnrate = calc_fnrate(pixeles_verdes,pixeles_amarillos)
recall = calc_recall(pixeles_amarillos,pixeles_verdes)
tnrate = calc_tnrate(pixeles_azules,pixeles_rojos)
fprate = calc_fprate(pixeles_rojos,pixeles_azules)
precision = calc_precision(pixeles_amarillos,pixeles_rojos)

t0 = time.time()
fscore_07= calc_fscores(precision,recall)
t1 = time.time()
total = t1-t0
print("El false negative rate para Caras07 es: ",fnrate)
print("El true postive rate o recall Caras07 es: ",recall)
print("El true negative rate para Caras07 es: ",tnrate)
print("El false positive rate para Caras07 es: ",fprate)
print("El precision para Caras07 es: ",precision)
print("El fscore para Caras07 es: ",fscore_07)
# ----------------------------------------------------------------------------------------------------------------------------------------
# promedio del índice F-Score 

fscore = (fscore_01 + fscore_02 + fscore_03 + fscore_04 + fscore_05 + fscore_06 + fscore_07)/7
print("El fscore para el algortimo es: ",fscore)
