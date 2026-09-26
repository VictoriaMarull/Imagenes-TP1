#PROBLEMA 1 

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen en escala de grises
img = cv2.imread(
    "Imagenes-TP1/Imagen_con_detalles_escondidos.tif",
    cv2.IMREAD_GRAYSCALE
)

print(img.shape) #Tamaño de la imagen.
print(img.dtype) #Tipo de dato.
print(img.min()) #Intensidad mínima.
print(img.max()) #Intensidad máxima.

plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Imagen original")
plt.axis("off")
plt.show()

# crear la función de ecualización local

def ecualizacion_local(img, M, N):

    # Verifico que la ventana tenga dimensiones impares para que haya un único píxel central
    if M % 2 == 0 or N % 2 == 0:
        print("M y N deben ser impares")
        return None

    filas, columnas = img.shape

    borde_filas = M // 2
    borde_columnas = N // 2

    # Agrego bordes para poder procesar los píxeles de los extremos
    img_borde = cv2.copyMakeBorder( # agrega píxeles alrededor de una imagen, para poder procesar también los píxeles que están en los bordes.
        img,
        borde_filas,
        borde_filas,
        borde_columnas,
        borde_columnas,
        cv2.BORDER_REPLICATE # indica que el borde agregado debe completarse repitiendo los valores de los píxeles que están en el límite de la imagen.
    )

    # Creo una imagen vacía con el mismo tamaño que la original
    img_salida = np.zeros_like(img) # crea una imagen nueva, llena de ceros —completamente negra—, con el mismo tamaño y tipo de dato que img.

    # Recorro todos los píxeles
    for fila in range(filas):
        for columna in range(columnas):

            # Extraigo la ventana local
            ventana = img_borde[
                fila:fila + M,
                columna:columna + N
            ]

            # Ecualizo solamente la ventana
            ventana_ecualizada = cv2.equalizeHist(ventana) # mejora el contraste de una imagen en escala de grises, redistribuyendo sus intensidades entre negro y blanco.

            # Guardo únicamente el píxel central ecualizado
            img_salida[fila, columna] = ventana_ecualizada[
                borde_filas,
                borde_columnas
            ]

    return img_salida

# probar diferentes ventanas

img_local_7 = ecualizacion_local(img, 7, 7) # Ventana pequeña: resalta detalles muy locales, pero genera más ruido.
img_local_15 = ecualizacion_local(img, 15, 15)
img_local_31 = ecualizacion_local(img, 31, 31) # Ventana mediana: permite ver bien los objetos sin producir tanto ruido
img_local_63 = ecualizacion_local(img, 63, 63) # Ventana grande: suaviza el resultado y pierde parte de la localidad.


plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Imagen original")
plt.axis("off")

plt.subplot(232)
plt.imshow(img_local_7, cmap="gray", vmin=0, vmax=255)
plt.title("Ventana 7 x 7")
plt.axis("off")

plt.subplot(233)
plt.imshow(img_local_15, cmap="gray", vmin=0, vmax=255)
plt.title("Ventana 15 x 15")
plt.axis("off")

plt.subplot(234)
plt.imshow(img_local_31, cmap="gray", vmin=0, vmax=255)
plt.title("Ventana 31 x 31")
plt.axis("off")

plt.subplot(235)
plt.imshow(img_local_63, cmap="gray", vmin=0, vmax=255)
plt.title("Ventana 63 x 63")
plt.axis("off")

plt.tight_layout()
plt.show()

