import cv2
import numpy as np
import os
# posicionarse en la carpeta frame
os.chdir("frames")

# Cargar la imagen
image = cv2.imread('frame_70.jpg')

# Convertir a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar un suavizado Gaussiano para reducir el ruido
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Aplicar filtro top-hat para resaltar regiones brillantes
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)

# Binarizar la imagen usando un umbral adaptativo
_, binary = cv2.threshold(tophat, 50, 255, cv2.THRESH_BINARY)

# Encontrar contornos para identificar posibles dados
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crear una máscara vacía
mask = np.zeros_like(binary)

# Filtrar contornos grandes (dados) y dibujarlos en la máscara
for contour in contours:
    area = cv2.contourArea(contour)
    if 300 < area < 2000:  # Ajustar según el tamaño esperado de los dados
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

# Aplicar la máscara a la imagen binarizada
masked_binary = cv2.bitwise_and(binary, binary, mask=mask)

# Usar HoughCircles para detectar puntos blancos en la máscara
circles = cv2.HoughCircles(
    tophat,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=10,  # Distancia mínima entre los centros de los círculos
    param1=50,   # Umbral del detector Canny
    param2=10,   # Umbral del acumulador para la detección de círculos
    minRadius=2, # Radio mínimo de los círculos
    maxRadius=5  # Radio máximo de los círculos
)

# Dibujar los círculos detectados en la imagen original
output = image.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        center = (circle[0], circle[1])  # Coordenadas del centro
        radius = circle[2]              # Radio del círculo
        # Dibujar el contorno del círculo
        cv2.circle(output, center, radius, (255, 0, 0), 2)

# Mostrar los resultados
cv2.imshow('Original con Círculos Detectados', output)
cv2.imshow('Máscara Final', masked_binary)
cv2.imshow('Filtro Top-Hat', tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()
