# PROBLEMA 2
import cv2
import numpy as np
import matplotlib.pyplot as plt

# hacer que el procesamiento devuelva resultados

def procesar_examen(numero_examen):

    ruta_imagen = (
        f"examen_{numero_examen}.png"
    )

    img = cv2.imread(
        ruta_imagen,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        print(f"No se pudo abrir el examen {numero_examen}")
        return None, 0

    # A: detectar las respuestas

    plt.imshow(img, cmap="gray")
    plt.title("Examen original")
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # Umbralizar 

    img_th = img < 200 # Los píxeles oscuros pasan a True y los píxeles blancos pasan a False. Separar lo que nos interesa del fondo las lineas y letras del fondo blanco.

    plt.imshow(img_th, cmap="gray")
    plt.title("Examen umbralizado")
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # sumar los píxeles por columnas y filas
    img_cols = np.sum(img_th, 0) # uma hacia abajo y produce un resultado para cada columna. Sirve para encontrar las líneas verticales.
    img_rows = np.sum(img_th, 1) # Suma horizontalmente y produce un resultado para cada fila. Sirve para encontrar las líneas horizontales.

    
    plt.subplot(2, 1, 1)
    plt.plot(img_cols)
    plt.title("Suma por columnas")

    plt.subplot(2, 1, 2)
    plt.plot(img_rows)
    plt.title("Suma por filas")

    plt.show()

    # umbralizar las sumas
    img_cols_th = img_cols > 300 # 300 separa las cuatro líneas verticales altas.
    img_rows_th = img_rows > 450 # 450 separa las líneas horizontales principales del texto.


    pos_columnas = np.where(img_cols > 300)[0]
    pos_filas = np.where(img_rows > 450)[0]

    print(pos_columnas)
    print(pos_filas)
    # obtener las posiciones de las líneas

    pos_columnas = np.where(img_cols_th)[0] # (.where) Devuelve las posiciones donde la condición vale True.
    pos_filas = np.where(img_rows_th)[0] 

    # vimos que las lineas ocupan más de un píxel

    # reunir las posiciones de una misma línea

    def unir_lineas(posiciones):

        lineas = []

        inicio = posiciones[0]
        anterior = posiciones[0]

        for posicion in posiciones[1:]:

            if posicion > anterior + 1:
                centro = int((inicio + anterior) / 2)
                lineas.append(centro)
                inicio = posicion

            anterior = posicion

        centro = int((inicio + anterior) / 2)
        lineas.append(centro)

        return np.array(lineas)

    lineas_verticales = unir_lineas(pos_columnas)
    lineas_horizontales = unir_lineas(pos_filas)

    print("Líneas verticales:")
    print(lineas_verticales)

    print("Líneas horizontales:")
    print(lineas_horizontales)

    # visualizar las líneas

    img_lineas = cv2.cvtColor( # cambiar el formato de color de una imagen.
        img,
        cv2.COLOR_GRAY2RGB
    )

    # Dibujamos las verticales
    for x in lineas_verticales:

        cv2.line(
            img_lineas,
            (x, 0),
            (x, img.shape[0] - 1), #( img.shape[0] devuelve alto) Devuelve el tamaño de la imagen.
            (255, 0, 0),
            1
        )
        
    # Dibujamos las horizontales
    for y in lineas_horizontales:

        cv2.line( # dibujar una línea sobre una imagen.
            img_lineas,
            (0, y),
            (img.shape[1] - 1, y),
            (255, 0, 0),
            1
        )
        
    plt.imshow(img_lineas)
    plt.title("Líneas detectadas")
    plt.show()

    # separar las celdas

    # Las líneas verticales
    x1 = lineas_verticales[0]
    x2 = lineas_verticales[1]
    x3 = lineas_verticales[2]
    x4 = lineas_verticales[3]

    # Las líneas horizontales
    y1 = lineas_horizontales[0]
    y2 = lineas_horizontales[1]
    y3 = lineas_horizontales[2]
    y4 = lineas_horizontales[3]
    y5 = lineas_horizontales[4]
    y6 = lineas_horizontales[5]

    #pata pronar celda 1
    celda_1 = img[y1 + 2:y2 - 2, x1 + 2:x2 - 2]

    plt.imshow(celda_1, cmap="gray")
    plt.title("Celda de la pregunta 1")
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # umbralizar una celda

    celda_th = np.uint8(celda_1 < 200) * 255 # Convierte el contenido oscuro en 255 y convierte el fondo blanco en 0.

    # buscar componentes conectadas

    num_labels, labels, stats, centroids = ( # num_labels: cantidad de componentes, incluido el fondo, labels: etiqueta asignada a cada píxel, stats: posición, ancho, alto y área de cada componente, centroids: centro de cada componente.
        cv2.connectedComponentsWithStats(
            celda_th,
            connectivity=8,
            ltype=cv2.CV_32S
        )
    )

    # eliminar componentes muy pequeñas

    ix_area = stats[:, -1] > 5 # conserva únicamente los elementos cuya área sea mayor que 5 píxeles.
    stats = stats[ix_area, :]
    centroids = centroids[ix_area, :] 

    # Separar las preguntas 1 a 5

    celdas = []

    for i in range(5):       # recorre los cinco espacios formados entre las seis líneas horizontales, 

        celda = img[
            lineas_horizontales[i] + 2:
            lineas_horizontales[i + 1] - 2,    # Las preguntas 1 a 5 se encuentran entre las dos primeras líneas verticales.

            lineas_verticales[0] + 2:
            lineas_verticales[1] - 2     # El + 2 y - 2 eliminan los bordes de la tabla.
        ]

        celdas.append(celda)

    # Separar las preguntas 6 a 10

    for i in range(5):

        celda = img[
            lineas_horizontales[i] + 2:
            lineas_horizontales[i + 1] - 2,

            lineas_verticales[2] + 2:
            lineas_verticales[3] - 2
        ]

        celdas.append(celda)
        
    # Mostrar las diez preguntas

    plt.figure(figsize=(12, 8))

    for i in range(10):

        plt.subplot(2, 5, i + 1)    # 2: dos filas. 5: cinco columnas. i + 1: posición del gráfico. Se suma 1 porque las posiciones de subplot comienzan en 1, aunque i comience en 0.
        plt.imshow(celdas[i], cmap="gray")
        plt.title(f"Pregunta {i + 1}")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

    # Umbralizar una celda
    # Empezamos probando con la pregunta 1

    celda_1 = celdas[0]      # Elementos oscuros → 255, blanco. Fondo → 0, negro.
    celda_1_th = np.uint8(celda_1 < 200) * 255

    plt.subplot(1, 2, 1)
    plt.imshow(celda_1, cmap="gray")
    plt.title("Pregunta 1")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(celda_1_th, cmap="gray")
    plt.title("Pregunta 1 umbralizada")
    plt.xticks([])
    plt.yticks([])

    plt.show()


    # usar solamente la mitad superior

    mitad = int(celda.shape[0] / 2)

    zona_superior = celda[:mitad, :]   # :mitad: desde la primera fila hasta la mitad. :: todas las columnas.

    zona_th = zona_superior < 200 # convertir la parte superior en binaria (True: píxel negro., False: píxel blanco.)

    # función para encontrar el renglón nos encontramos el segmento horizontal mas largo

    def detectar_renglon(img_binaria):

        largo_mayor = 0
        fila_linea = 0
        inicio_linea = 0
        fin_linea = 0

        for fila in range(5, img_binaria.shape[0]): # al pregunta 9 esta mas arriba que las demas y nos salio el recorte mal cambiamos el 20 por 5 porwue al empezara buscar de la fila 20 no hagaraba  la letra 

            inicio_actual = 0
            largo_actual = 0

            for columna in range(img_binaria.shape[1]):

                if img_binaria[fila, columna]:

                    if largo_actual == 0:
                        inicio_actual = columna

                    largo_actual += 1

                    if largo_actual > largo_mayor:
                        largo_mayor = largo_actual
                        fila_linea = fila
                        inicio_linea = inicio_actual
                        fin_linea = columna

                else:

                    largo_actual = 0

        return fila_linea, inicio_linea, fin_linea
     

    # llamar la función

    fila_linea, inicio_linea, fin_linea = detectar_renglon(zona_th)

    print("Fila del renglón:", fila_linea)
    print("Inicio:", inicio_linea)
    print("Fin:", fin_linea)

    # recortar arriba del renglón

    inicio_fila = fila_linea - 15  # cambiamos 25 a 15 porque tomaba las letras de arriba el recorte aun 

    if inicio_fila < 0:
        inicio_fila = 0
        
    respuesta = zona_superior[
        inicio_fila:fila_linea,
        inicio_linea:fin_linea + 1
    ]

    plt.imshow(respuesta, cmap="gray")
    plt.title("Respuesta recortada")
    plt.xticks([])
    plt.yticks([])
    plt.show()


    # umbralizar el recorte donde quedó la respuesta

    respuesta_th = np.uint8(respuesta < 200) * 255

    plt.imshow(respuesta_th, cmap="gray")
    plt.title("Respuesta umbralizada")
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # lo hacemos para todas las preguntas

    respuestas = []
    respuestas_th = []

    for i in range(10):

            # Selecciono una celda
            celda = celdas[i]

            # Me quedo con la mitad superior
            mitad = int(celda.shape[0] / 2)
            zona_superior = celda[:mitad, :]

            # Umbralizo para buscar el renglón
            zona_th = zona_superior < 200

            # Detecto el renglón
            fila_linea, inicio_linea, fin_linea = detectar_renglon(zona_th)

            # Subo 15 píxeles desde el renglón
            inicio_fila = fila_linea - 15

            if inicio_fila < 0:
                inicio_fila = 0

            # Recorto solamente la zona de la respuesta
            respuesta = zona_superior[
                inicio_fila:fila_linea,
                inicio_linea:fin_linea + 1
            ]

            # Umbralizo la respuesta
            respuesta_th = np.uint8(respuesta < 200) * 255

            # Guardo ambos resultados
            respuestas.append(respuesta)
            respuestas_th.append(respuesta_th)
        

    plt.figure(figsize=(14, 6))

    for i in range(10):

        plt.subplot(2, 5, i + 1)

        plt.imshow(
            respuestas_th[i],
            cmap="gray"
        )

        plt.title(f"Respuesta {i + 1}")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

    # Creamos una función que recibe cada imagen de respuestas_th y devuelve A, B, C o D

    def identificar_letra(img_letra):

        # Verifico si la respuesta está vacía
        cantidad_pixeles = np.sum(img_letra > 0)

        if cantidad_pixeles < 15:
            return "SIN RESPUESTA"
        
        
        # Buscar los componentes blancos
        cantidad, etiquetas, estadisticas, centroides = \
            cv2.connectedComponentsWithStats(
                np.uint8(img_letra > 0),
                8
            )
        
        # Si no encontró ninguna letra
        if cantidad <= 1:
            return "SIN RESPUESTA" 
        
        # Obtener las áreas sin contar el fondo
        areas = estadisticas[1:, cv2.CC_STAT_AREA]

        # Considerar solamente componentes con más de 15 píxeles
        componentes_validos = areas > 15

        # Contar cuántos componentes grandes hay
        cantidad_letras = np.sum(componentes_validos)

        # No hay ninguna letra
        if cantidad_letras == 0:
            return "SIN RESPUESTA"

        # Hay más de una letra
        if cantidad_letras > 1:
            return "RESPUESTA MULTIPLE"

        # Obtener la posición del único componente válido
        posiciones_validas = np.where(componentes_validos)[0]

        # Se suma 1 porque el componente 0 es el fondo
        componente_mayor = posiciones_validas[0] + 1

        # Conservar solamente la letra
        img_letra = np.uint8(
            etiquetas == componente_mayor
        ) * 255

        
        # Recortar el espacio negro y conservar solamente la letra # ya que sin este pedazo de codigo nos soluciono que detectabamal la D la consideraba de la A
        filas, columnas = np.where(img_letra > 0)

        fila_inicio = filas.min()
        fila_fin = filas.max()
        columna_inicio = columnas.min()
        columna_fin = columnas.max()

        img_letra = img_letra[
            fila_inicio:fila_fin + 1,
            columna_inicio:columna_fin + 1
        ]

        # Encuentro los contornos y su jerarquía
        contornos, jerarquia = cv2.findContours(  # aca cuanta los agujeros que tiene la letra y devuelve la letra correspondiente
            img_letra,
            cv2.RETR_TREE,  # cv2.RETR_TREE también guarda la relación entre ellos. Si un contorno está dentro de otro, puede representar un agujero.
            cv2.CHAIN_APPROX_NONE
        )

        if jerarquia is None:
            return "SIN RESPUESTA"

        cantidad_agujeros = 0
        
        for i in range(len(contornos)):
            if jerarquia[0, i, 3] != -1:
                cantidad_agujeros += 1


        if cantidad_agujeros == 2:
            return "B"

        elif cantidad_agujeros == 0:
            return "C"

        elif cantidad_agujeros == 1:

            suma_columnas = np.sum(img_letra > 0, axis=0)

            mitad = int(img_letra.shape[1] / 2)
            lado_izquierdo = suma_columnas[:mitad]

            maximo_izquierdo = lado_izquierdo.max()

            if maximo_izquierdo > img_letra.shape[0] * 0.7:
                return "D"
            else:
                return "A"

        else:
            return "NO IDENTIFICADA"
        
        #Aplicar la función a las diez preguntas
    respuestas_detectadas = []

    for i in range(10):

        letra = identificar_letra(
            respuestas_th[i]
        )

        respuestas_detectadas.append(letra)

        print(
            f"Pregunta {i + 1}: {letra}"
        )


    # Crear una figura
    plt.figure(figsize=(14, 6))

    for i in range(10):

        plt.subplot(2, 5, i + 1)

        plt.imshow(
            respuestas_th[i],
            cmap="gray"
        )

        plt.title(
            f"Pregunta {i + 1}: "
            f"{respuestas_detectadas[i]}"
        )

        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()
        
    # Guardar las respuestas correctas
    respuestas_correctas = [
        "C", "B", "A", "D", "B",
        "B", "A", "B", "D", "D"
    ]

    # Comparar las respuestas
    cantidad_correctas = 0

    for i in range(10):

        if respuestas_detectadas[i] == respuestas_correctas[i]:

            print(f"Pregunta {i + 1}: OK")
            cantidad_correctas += 1

        else:

            print(f"Pregunta {i + 1}: MAL")
            
    # Mostrar también qué respondió

    cantidad_correctas = 0

    for i in range(10):

        respuesta_alumno = respuestas_detectadas[i]
        respuesta_correcta = respuestas_correctas[i]

        if respuesta_alumno == respuesta_correcta:

            estado = "OK"
            cantidad_correctas += 1

        else:

            estado = "MAL"

        print(
            f"Pregunta {i + 1}: {estado} "
            f"| Alumno: {respuesta_alumno} "
            f"| Correcta: {respuesta_correcta}"
        )
        
    print(
        f"Resultado final: "
        f"{cantidad_correctas}/10"
    )
        
    if cantidad_correctas >= 6:

        print("APROBADO")

    else:

        print("DESAPROBADO")
        

    # B: validar los datos del encabezado

    # quedarnos con el encabezado

    # El encabezado termina donde comienza la tabla de preguntas

    encabezado = img[
        :lineas_horizontales[0],
        :
    ]

    # Umbralizar el encabezado
    encabezado_th = np.uint8(
        encabezado < 200
    ) * 255


    plt.imshow(encabezado_th, cmap="gray")
    plt.title("Encabezado umbralizado")
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # encontrar la fila de los renglones

    suma_filas_encabezado = np.sum(
        encabezado_th > 0,
        axis=1
    )

    fila_renglones = np.argmax(
        suma_filas_encabezado
    )

    suma_filas_encabezado = np.sum(
        encabezado_th > 0,
        axis=1
    )

    fila_renglones = np.argmax(
        suma_filas_encabezado
    )

    plt.plot(suma_filas_encabezado)
    plt.title("Suma de píxeles por filas del encabezado")
    plt.show()


    # detectando los tres renglones 

    fila_binaria = encabezado_th[
        fila_renglones,
        :
    ] > 0


    def detectar_segmentos(fila_binaria):

        segmentos = []

        inicio = 0
        largo = 0

        for columna in range(len(fila_binaria)):

            if fila_binaria[columna]:

                if largo == 0:
                    inicio = columna

                largo += 1

            else:

                if largo > 20:

                    segmentos.append(
                        [inicio, columna - 1]
                    )

                largo = 0

        # Por si el último segmento llega hasta el final
        if largo > 20:     # Conservar solamente líneas que midan más de 20 píxeles.

            segmentos.append(
                [inicio, len(fila_binaria) - 1]
            )

        return segmentos

    segmentos = detectar_segmentos(
        fila_binaria
    )

    print("Segmentos encontrados:")
    print(segmentos)

    segmentos[0]  # Name
    segmentos[1]  # Date
    segmentos[2]  # Class

    # recortar el texto de cada campo

    campos = []

    altura_texto = fila_renglones

    for inicio, fin in segmentos:

        campo = encabezado[
            0:fila_renglones,
            inicio:fin + 1
        ]

        campos.append(campo)
        
    campo_name = campos[0]
    campo_date = campos[1]
    campo_class = campos[2]


    titulos = [
        "Name",
        "Date",
        "Class"
    ]

    plt.figure(figsize=(12, 3))

    for i in range(3):

        plt.subplot(1, 3, i + 1)
        plt.imshow(campos[i], cmap="gray")
        plt.title(titulos[i])
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

    # detectar y encuadrar caracteres

    def analizar_campo(campo):

        # Umbralizar
        campo_th = np.uint8(
            campo < 200
        ) * 255

        # Buscar componentes conectados
        cantidad, etiquetas, stats, centroides = \
            cv2.connectedComponentsWithStats(
                campo_th,
                connectivity=8,
                ltype=cv2.CV_32S
            )

        # Eliminar el fondo
        stats = stats[1:, :]

        # Eliminar componentes pequeños
        areas = stats[:, cv2.CC_STAT_AREA]
        componentes_validos = areas > 5
        stats = stats[componentes_validos, :]

        # Ordenar los rectángulos de izquierda a derecha
        orden = stats[:, 0].argsort()
        stats = stats[orden, :]

        # Convertir a color para dibujar rectángulos
        campo_color = cv2.cvtColor(
            campo,
            cv2.COLOR_GRAY2RGB
        )

        # Dibujar un rectángulo por carácter
        for st in stats:

            x = st[cv2.CC_STAT_LEFT]
            y = st[cv2.CC_STAT_TOP]
            ancho = st[cv2.CC_STAT_WIDTH]
            alto = st[cv2.CC_STAT_HEIGHT]

            cv2.rectangle(
                campo_color,
                (x, y),
                (x + ancho, y + alto),
                (255, 0, 0),
                1
            )
            
    # contar caracteres y palabras

        # Cantidad de rectángulos = cantidad de caracteres
        cantidad_caracteres = len(stats)

        # Si está vacío, tiene cero palabras
        if cantidad_caracteres == 0:

            cantidad_palabras = 0

        else:

            # Si hay caracteres, existe al menos una palabra
            cantidad_palabras = 1

            # Comparar cada carácter con el anterior
            for i in range(1, len(stats)):

                x_anterior = stats[
                    i - 1,
                    cv2.CC_STAT_LEFT
                ]

                ancho_anterior = stats[
                    i - 1,
                    cv2.CC_STAT_WIDTH
                ]

                x_actual = stats[
                    i,
                    cv2.CC_STAT_LEFT
                ]

                # Espacio entre dos caracteres
                espacio = (
                    x_actual
                    - (x_anterior + ancho_anterior)
                )

                # Espacio grande: comienza otra palabra
                if espacio > 5:
                    cantidad_palabras += 1

        return (
            cantidad_caracteres,
            cantidad_palabras,
            campo_color
        )
        
    # analizar los tres campos

    caracteres_name, palabras_name, name_rectangulos = \
        analizar_campo(campo_name)

    caracteres_date, palabras_date, date_rectangulos = \
        analizar_campo(campo_date)

    caracteres_class, palabras_class, class_rectangulos = \
        analizar_campo(campo_class)
        
    imagenes_campos = [
        name_rectangulos,
        date_rectangulos,
        class_rectangulos
    ]

    plt.figure(figsize=(12, 3))

    for i in range(3):

        plt.subplot(1, 3, i + 1)
        plt.imshow(imagenes_campos[i])
        plt.title(titulos[i])
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

    # validar Name, Date y Class

    # Name

    if palabras_name >= 2 and caracteres_name <= 25:
        estado_name = "OK"
    else:
        estado_name = "MAL"
        
    # Date
    if palabras_date == 1 and caracteres_date == 8:
        estado_date = "OK"
    else:
        estado_date = "MAL"

    # Class
    if caracteres_class == 1:
        estado_class = "OK"
    else:
        estado_class = "MAL"
        
    print(
        f"Name: {estado_name} "
        f"| Palabras: {palabras_name} "
        f"| Caracteres: {caracteres_name}"
    )

    print(
        f"Date: {estado_date} "
        f"| Palabras: {palabras_date} "
        f"| Caracteres: {caracteres_date}"
    )

    print(
        f"Class: {estado_class} "
        f"| Caracteres: {caracteres_class}"
    )

    # D

    return campo_name, cantidad_correctas

# procesar los cinco exámenes

nombres = []
notas = []

for numero_examen in range(1, 6):

    campo_name, cantidad_correctas = \
        procesar_examen(numero_examen)

    if campo_name is not None:
        nombres.append(campo_name)
        notas.append(cantidad_correctas)
    
# calcular el tamaño de la imagen final
# Como los recortes de los nombres pueden tener tamaños diferentes

margen = 20

alto_mayor = 0
ancho_mayor = 0

for nombre in nombres:

    alto = nombre.shape[0]
    ancho = nombre.shape[1]

    if alto > alto_mayor:
        alto_mayor = alto

    if ancho > ancho_mayor:
        ancho_mayor = ancho

# crear una imagen blanca vacía

alto_fila = alto_mayor + 2 * margen

alto_salida = alto_fila * len(nombres)
ancho_salida = ancho_mayor + 250

imagen_salida = np.ones(
    (
        alto_salida,
        ancho_salida,
        3
    ),
    dtype=np.uint8
) * 255

# colocar cada nombre en la imagen

for i in range(len(nombres)):

    nombre = nombres[i]

    # Convertir el recorte a color
    nombre_color = cv2.cvtColor(
        nombre,
        cv2.COLOR_GRAY2BGR
    )

    alto = nombre.shape[0]
    ancho = nombre.shape[1]

    x = margen
    y = i * alto_fila + margen

    # Copiar el nombre en la imagen final
    imagen_salida[
        y:y + alto,
        x:x + ancho
    ] = nombre_color
    
# diferenciar aprobados y desaprobados

    if notas[i] >= 6:

        estado = "APROBADO"
        color = (0, 255, 0)  # es verde.

    else:

        estado = "DESAPROBADO"
        color = (0, 0, 255) # es rojo.
        

# dibujar el rectángulo

    cv2.rectangle(
        imagen_salida,
        (x, y),
        (x + ancho, y + alto),
        color,
        2
    )
    
# escribir el estado

    cv2.putText(
        imagen_salida,
        estado,
        (ancho_mayor + 40, y + alto - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

# mostrar la imagen final

imagen_salida_rgb = cv2.cvtColor(
    imagen_salida,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(10, 8))
plt.imshow(imagen_salida_rgb)
plt.title("Resultados de los exámenes")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

# guardar la imagen final

cv2.imwrite(
    "resultados_examenes.png",
    imagen_salida
)
