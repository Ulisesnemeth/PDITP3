import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir("frames")

# Cargar la imagen
image = cv2.imread('frame_70.jpg')
if image is None:
    print("No se encontró la imagen. Asegúrate de que 'frame_70.jpg' exista en el directorio 'frames'.")
    exit()

# Convertir la imagen de BGR a HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir el color objetivo (según tu descripción)
target_color_hsv = np.array([1, 31, 220])

# Definir un delta más pequeño para hacer la máscara más precisa
delta = np.array([10, 50, 50])  # Delta más pequeño para el tono y la saturación, más grande para el valor

# Calcular los límites inferior y superior para la máscara
lower_limit = np.clip(target_color_hsv - delta, [0, 0, 0], [179, 255, 255])
upper_limit = np.clip(target_color_hsv + delta, [0, 0, 0], [179, 255, 255])

# Depuración: imprimir los límites inferior y superior para asegurarse de que estén dentro del rango
print("Color objetivo HSV:", target_color_hsv)
print("Límite inferior:", lower_limit)
print("Límite superior:", upper_limit)

# Crear una máscara para el rango de color especificado
mask = cv2.inRange(image_hsv, lower_limit, upper_limit)

# Aplicar la máscara a la imagen
result = cv2.bitwise_and(image, image, mask=mask)

# Función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

# Mostrar las imágenes originales y procesadas
imshow(image_hsv, title='Imagen Original HSV', color_img=False)
imshow(image, title='Imagen Original', color_img=True)
imshow(mask, title='Máscara (Colores Permitidos)', color_img=False)
imshow(result, title='Resultado Final', color_img=True)
