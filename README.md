# Trabajo Práctico N°1 — Procesamiento Avanzado de Imágenes

Proyecto desarrollado para la materia Procesamiento Avanzado de Imágenes de la Licenciatura en Ciencia de Datos.

El trabajo práctico aborda dos problemas diferentes de procesamiento digital de imágenes utilizando Python, OpenCV, NumPy y Matplotlib:

1. Ecualización local de histograma.
2. Corrección automática de exámenes multiple choice.

En ambos casos se busca obtener información directamente a partir del contenido de las imágenes, evitando depender de coordenadas absolutas previamente definidas.

---

## Integrantes

- Victoria Marull
- Inés Medina
- Paz Sevilla

Profesor: Gonzalo Daniel Sad

---

## Tecnologías utilizadas

- Python
- OpenCV
- NumPy
- Matplotlib

---

## Instalación

Para ejecutar el proyecto es necesario tener Python instalado.

Las librerías utilizadas pueden instalarse con:

```bash
pip install opencv-python numpy matplotlib
```

---

## Estructura general del proyecto

```text
.
├── Problema_1.py
├── Problema_2.py
├── Imagenes-TP1/
│   ├── examen_1.png
│   ├── examen_2.png
│   ├── examen_3.png
│   ├── examen_4.png
│   └── examen_5.png
├── TP_prueba/
│   └── Imagen_con_detalles_escondidos.tif
└── README.md
```

El segundo problema genera además una imagen de salida:

```text
resultados_examenes.png
```

Los nombres de las carpetas deben coincidir con las rutas utilizadas dentro de los scripts.

---

# Problema 1 — Ecualización local de histograma

## Objetivo

El primer problema consiste en mejorar el contraste de una imagen que contiene diferentes objetos con intensidades muy similares a las de sus fondos locales.

Una ecualización global no resulta adecuada en este caso, ya que utiliza el histograma completo de la imagen y puede perder información correspondiente a zonas particulares.

Por este motivo, se implementó una ecualización local de histograma, donde la transformación se calcula utilizando una ventana que se desplaza píxel a píxel.

---

## Funcionamiento

La imagen se carga inicialmente en escala de grises.

La función principal recibe:

```python
ecualizacion_local(img, M, N)
```

donde:

- `img`: imagen de entrada.
- `M`: cantidad de filas de la ventana.
- `N`: cantidad de columnas de la ventana.

Las dimensiones de la ventana deben ser impares para garantizar la existencia de un único píxel central.

---

## Procesamiento de los bordes

Cuando la ventana se encuentra cerca de los límites de la imagen, parte de ella quedaría fuera de la región disponible.

Para evitar este problema se utiliza:

```python
cv2.copyMakeBorder()
```

junto con:

```python
cv2.BORDER_REPLICATE
```

De esta manera, los valores de los píxeles ubicados en los bordes se replican y es posible procesar todos los píxeles de la imagen.

---

## Ecualización local

Para cada píxel de la imagen:

1. Se extrae una ventana local de tamaño `M x N`.
2. Se aplica:

```python
cv2.equalizeHist()
```

3. Se obtiene el valor ecualizado correspondiente al píxel central.
4. Ese valor se coloca en la misma posición de la imagen de salida.

El procedimiento se repite hasta recorrer toda la imagen.

---

## Tamaños de ventana analizados

Se probaron cuatro tamaños:

```text
7 x 7
15 x 15
31 x 31
63 x 63
```

El tamaño de la ventana afecta directamente el resultado.

### Ventana 7 x 7

Realiza un procesamiento muy localizado.

Permite detectar pequeñas variaciones de intensidad, aunque también amplifica considerablemente el ruido.

### Ventana 15 x 15

Mantiene un análisis local, pero reduce parte del ruido producido por ventanas más pequeñas.

### Ventana 31 x 31

Presenta un buen equilibrio entre recuperación de detalles y estabilidad de la imagen.

### Ventana 63 x 63

Considera regiones más grandes de la imagen, por lo que comienza a perder localidad y produce un resultado más cercano a una ecualización global.

---

## Detalles encontrados

Mediante la ecualización local fue posible identificar cinco elementos que inicialmente presentaban muy poco contraste respecto de sus fondos:

- Un cuadrado en la zona superior izquierda.
- Una línea diagonal en la zona superior derecha.
- La letra `a` en el centro.
- Varias líneas horizontales en la zona inferior izquierda.
- Una figura circular en la zona inferior derecha.

---

# Problema 2 — Corrección automática de exámenes

## Objetivo

El segundo problema consiste en desarrollar un sistema capaz de corregir automáticamente imágenes de exámenes multiple choice.

Cada examen contiene:

- 10 preguntas.
- 4 opciones posibles: A, B, C y D.
- Un encabezado con:
  - Name
  - Date
  - Class

El sistema debe procesar solamente la imagen del examen, sin recibir las coordenadas de las preguntas como entrada.

---

## Respuestas correctas

La clave utilizada para corregir los exámenes es:

```text
1. C
2. B
3. A
4. D
5. B
6. B
7. A
8. B
9. D
10. D
```

Una pregunta se considera incorrecta si:

- La opción elegida no coincide con la respuesta correcta.
- No existe respuesta.
- Se detecta más de una respuesta.

Para aprobar el examen se necesitan al menos:

```text
6 respuestas correctas
```

---

## Flujo general del procesamiento

El procesamiento realizado puede resumirse de la siguiente manera:

```text
Imagen original
      ↓
Escala de grises
      ↓
Umbralización
      ↓
Detección de líneas
      ↓
Separación de preguntas
      ↓
Localización de respuestas
      ↓
Segmentación de letras
      ↓
Identificación A / B / C / D
      ↓
Comparación con respuestas correctas
      ↓
Cálculo de nota
      ↓
APROBADO / DESAPROBADO
```

Paralelamente se procesa el encabezado:

```text
Encabezado
    ↓
Detección de renglones
    ↓
Separación de Name / Date / Class
    ↓
Componentes conectadas
    ↓
Conteo de caracteres y palabras
    ↓
Validación de campos
```

---

## 1. Lectura y umbralización

Cada examen se carga en escala de grises utilizando OpenCV.

Luego se realiza una umbralización:

```python
img_th = img < 200
```

Los píxeles oscuros correspondientes a líneas, letras y respuestas quedan separados del fondo blanco.

---

## 2. Detección automática de la tabla

Para evitar utilizar coordenadas fijas, se calcula la cantidad de píxeles oscuros presentes en cada fila y columna.

```python
img_cols = np.sum(img_th, 0)
img_rows = np.sum(img_th, 1)
```

Las líneas verticales y horizontales de la tabla contienen una cantidad mucho mayor de píxeles oscuros que el resto de la imagen.

Se utilizan diferentes umbrales para detectarlas:

```python
img_cols_th = img_cols > 300
img_rows_th = img_rows > 450
```

---

## 3. Agrupamiento de líneas

Las líneas de la tabla pueden ocupar varios píxeles de ancho.

Por este motivo, varias posiciones consecutivas pueden corresponder a una misma línea.

La función:

```python
unir_lineas()
```

agrupa estas posiciones y utiliza el punto central de cada grupo como posición representativa.

---

## 4. Separación de las preguntas

A partir de las líneas detectadas se obtienen las celdas correspondientes a las diez preguntas.

Las preguntas 1 a 5 se encuentran en la columna izquierda y las preguntas 6 a 10 en la columna derecha.

Al realizar los recortes se eliminan algunos píxeles de los bordes para evitar que las líneas de la tabla interfieran con la detección de las respuestas.

---

## 5. Localización de la respuesta

Dentro de cada celda se analiza principalmente la mitad superior, donde se encuentra el espacio destinado a completar la respuesta.

El algoritmo busca automáticamente el renglón horizontal de respuesta.

Una vez localizado, se recorta la zona inmediatamente superior para conservar únicamente la letra escrita por el alumno.

La región obtenida vuelve a umbralizarse antes de analizarla.

---

## 6. Componentes conectadas

Para analizar la respuesta se utiliza:

```python
cv2.connectedComponentsWithStats()
```

Esta función permite detectar los elementos independientes presentes en una imagen binaria.

Las componentes muy pequeñas se eliminan para reducir el ruido.

Si no se encuentra ninguna componente suficientemente grande:

```text
SIN RESPUESTA
```

Si se detecta más de una componente válida:

```text
RESPUESTA MULTIPLE
```

En ambos casos la pregunta se considera incorrecta.

---

## 7. Identificación de las letras

El reconocimiento de A, B, C y D no se realiza utilizando OCR.

En cambio, se utilizan características geométricas de las letras.

La imagen se recorta alrededor de la componente detectada y se analizan sus contornos mediante:

```python
cv2.findContours()
```

### Letra B

La letra B posee dos regiones internas.

```text
Cantidad de agujeros = 2
```

Por lo tanto se identifica como B.

### Letra C

La letra C no posee regiones internas cerradas.

```text
Cantidad de agujeros = 0
```

Por lo tanto se identifica como C.

### Letras A y D

Las letras A y D poseen una región interna.

```text
Cantidad de agujeros = 1
```

Para diferenciarlas se analiza la mitad izquierda de la imagen de la letra.

Si existe una línea vertical suficientemente extensa, se identifica como D.

En caso contrario, se identifica como A.

---

## 8. Corrección automática

Las respuestas detectadas se comparan con:

```python
respuestas_correctas = [
    "C", "B", "A", "D", "B",
    "B", "A", "B", "D", "D"
]
```

Para cada pregunta se muestra un resultado similar a:

```text
Pregunta 1: OK
Pregunta 2: MAL
...
Pregunta 10: OK
```

Además, se informa:

- Respuesta del alumno.
- Respuesta correcta.
- Estado de la pregunta.

Finalmente se calcula la cantidad total de respuestas correctas.

---

## 9. Condición de aprobación

La condición utilizada es:

```python
if cantidad_correctas >= 6:
    print("APROBADO")
else:
    print("DESAPROBADO")
```

---

# Validación del encabezado

Además de corregir las respuestas, el sistema analiza los campos:

- Name
- Date
- Class

La región del encabezado se obtiene utilizando la primera línea horizontal de la tabla como referencia.

---

## Detección de campos

Dentro del encabezado se busca la fila correspondiente a los renglones donde se escriben los datos.

Después se detectan los tres segmentos horizontales correspondientes a:

```text
Name
Date
Class
```

Cada región se recorta automáticamente para ser analizada de manera independiente.

---

## Análisis de caracteres

Los caracteres se detectan mediante componentes conectadas.

Luego:

- Se eliminan componentes pequeñas.
- Se ordenan los caracteres de izquierda a derecha.
- Se cuenta la cantidad de caracteres.
- Se analiza la distancia entre caracteres consecutivos.

Una separación suficientemente grande entre dos caracteres se interpreta como un espacio entre palabras.

---

## Condiciones de validación

### Name

Debe cumplir:

```text
Al menos 2 palabras
Máximo 25 caracteres
```

### Date

Debe contener:

```text
8 caracteres
1 única palabra
```

### Class

Debe contener:

```text
1 único carácter
```

Cada campo se informa finalmente como `OK` o `MAL`.

---

# Exámenes analizados

Se procesaron cinco imágenes:

```text
examen_1.png
examen_2.png
examen_3.png
examen_4.png
examen_5.png
```

Los resultados obtenidos fueron:

| Alumno | Respuestas correctas | Resultado |
|---|---:|---|
| ESTEBANALVAREZ | 0/10 | DESAPROBADO |
| MARIA | 4/10 | DESAPROBADO |
| MARIA LOPEZ | 10/10 | APROBADO |
| LUCAS FERNANDEZ | 0/10 | DESAPROBADO |
| JUAN PEREZ | 10/10 | APROBADO |

---

# Validación de encabezados

| Alumno | Name | Date | Class |
|---|---|---|---|
| ESTEBANALVAREZ | MAL | OK | OK |
| MARIA | MAL | OK | OK |
| MARIA LOPEZ | OK | OK | OK |
| LUCAS FERNANDEZ | OK | MAL | OK |
| JUAN PEREZ | OK | OK | OK |

---

# Imagen final

Después de procesar todos los exámenes se genera automáticamente:

```text
resultados_examenes.png
```

La imagen contiene los recortes correspondientes al campo `Name` de cada alumno.

Los resultados se diferencian visualmente utilizando:

```text
Verde → APROBADO
Rojo  → DESAPROBADO
```

Además, se agrega el texto correspondiente al estado de cada examen.

---

# Ejecución

## Problema 1

Desde la carpeta principal del proyecto:

```bash
python Problema_1.py
```

El programa muestra:

- Imagen original.
- Resultado con ventana 7 x 7.
- Resultado con ventana 15 x 15.
- Resultado con ventana 31 x 31.
- Resultado con ventana 63 x 63.

---

## Problema 2

Ejecutar:

```bash
python Problema_2.py
```

El script procesa automáticamente los cinco exámenes y muestra las distintas etapas del procesamiento.

También informa en consola:

- Respuesta detectada.
- Respuesta correcta.
- Estado de cada pregunta.
- Cantidad de respuestas correctas.
- Resultado final.
- Validación de Name.
- Validación de Date.
- Validación de Class.

Finalmente genera:

```text
resultados_examenes.png
```

---

# Técnicas de procesamiento de imágenes utilizadas

A lo largo del trabajo se utilizaron diferentes técnicas:

- Conversión a escala de grises.
- Umbralización.
- Ecualización de histograma.
- Ecualización local.
- Replicación de bordes.
- Proyecciones por filas.
- Proyecciones por columnas.
- Segmentación de regiones.
- Componentes conectadas.
- Filtrado por área.
- Detección de contornos.
- Análisis de jerarquía de contornos.
- Recorte automático de regiones.
- Análisis geométrico de caracteres.

---

# Principales desafíos

Uno de los principales desafíos del segundo problema fue detectar automáticamente la estructura del examen sin utilizar coordenadas fijas.

La solución se basó en aprovechar las líneas horizontales y verticales del formulario como referencias.

También fue necesario evitar que estas líneas fueran detectadas como parte de las respuestas, por lo que los recortes se realizaron dejando pequeños márgenes respecto de los bordes.

Otro desafío fue reconocer las letras sin utilizar OCR. Para resolverlo se utilizaron características geométricas como:

- Número de regiones internas.
- Distribución de píxeles.
- Presencia de líneas verticales.

Finalmente, fue necesario ajustar diferentes umbrales y tamaños de recorte para obtener un procesamiento correcto de los distintos exámenes.

---

# Conclusión

Este trabajo permitió aplicar diferentes técnicas de procesamiento digital de imágenes sobre dos problemas concretos.

En el primer problema, la ecualización local permitió recuperar detalles que tenían intensidades muy similares a sus respectivos fondos y que no podían distinguirse correctamente mediante una transformación global.

En el segundo problema, la combinación de umbralización, proyecciones, componentes conectadas y contornos permitió desarrollar un sistema capaz de localizar automáticamente las preguntas de un examen, identificar las respuestas, corregirlas, validar la información del encabezado y determinar si cada alumno aprobó o desaprobó.

El enfoque utilizado permitió resolver ambos problemas trabajando principalmente a partir de la información contenida en las propias imágenes.