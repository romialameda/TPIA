# Trabajo Práctico Integrador - Otoño 2026

**Materia:** Inteligencia Artificial - UTN FRC

**Cuatrimestre:** 1° Cuatrimestre

**Año:** 2026

**Grupo:** 8

## Integrantes

Figueroa Prada, Juan Uriel – 407213

Gomez Alameda, Romina Abigail – 85296

Saggiorato, Gina – 95794

Storello Chiofalo, Juan Ignacio – 85408

</br>

# Exploración del Dataset (EDA)

## Estadísticas básicas

El dataset utilizado corresponde a Pascal VOC 2012 y contiene un total de **17.125 imágenes**. Para realizar el análisis exploratorio se tomó una muestra aleatoria de 500 imágenes, con el objetivo de reducir los tiempos de procesamiento manteniendo una representación adecuada del conjunto.
Soobre la muestra se calcularon estadísticas descriptivas relacionadas con dimensiones y tamaño de archivo.

Las imágenes presentan un ancho promedio de **464,3 píxeles** y una altura promedio de **391,2 píxeles**. El tamaño promedio de los archivos es de **109,1 KB**, con dimensiones mínimas de **200 × 111 píxeles** y máximas de **500 × 500 píxeles**.
Además, todas las imágenes analizadas presentan **3 canales de color (RGB)**, por lo que el dataset está compuesto íntegramente por imágenes en color.


| Métrica           | Ancho (px) | Alto (px) | Tamaño (KB) |
|------------------:|----------: |----------:|-----------: |
| Promedio          | 464.3      | 391.2     | 109.1       |
| Desvío estándar   | 63.2       | 66.0      | 42.5        |
| Mínimo            | 200        | 111       | 16.0        |
| Mediana           | 500        | 375       | 112.2       |
| Máximo            | 500        | 500       | 242.7       |
</br>

## Distribución por clases

Las anotaciones del dataset fueron utilizadas únicamente con fines estadísticos. Para optimizar el procesamiento, cada clase fue contabilizada una sola vez por imagen, independientemente de la cantidad de objetos presentes. De esta forma, si una imagen contiene múltiples instancias de una misma clase (por ejemplo, tres personas), la clase se contabiliza una sola vez para esa imagen.

La clase más frecuente es **person**, presente en 9.583 imágenes. Le siguen **chair** (1.366), **dog** (1.341), **car** (1.284) y **cat** (1.128). Las clases menos representadas son **cow** (340), **sheep** (357) y **bus** (467).

Estos resultados evidencian un desbalance importante entre las distintas categorías del conjunto de datos.

## Visualizaciones

Se generaron distintas visualizaciones para analizar las características del dataset, incluyendo un gráfico de dispersión de dimensiones, histogramas de tamaño de archivo y gráficos de barras para la distribución por clases.

![Visualizaciones explorativas del dataset](./images/visualizaciones_expoloracion_dataset.png)

Con estas imágenes podemos observar mejor el desbalance entre las clases, siendo *person* la categoría dominante en el datset, ocupando más de la mitad de este.

Además, se seleccionó una muestra aleatoria de imágenes con el objetivo de inspeccionar visualmente la variedad de escenas y objetos presentes en el conjunto de datos.

![Imagenes aleatorias del dataset](./images/img_aleatorias_exploracion.png)

</br>

# Embeddings con CLIP + Índice FAISS

## Carga del modelo CLIP

Se utilizó CLIP (ViT-B/32), un modelo capaz de representar imágenes y texto dentro de un mismo espacio semántico de 512 dimensiones.

El modelo fue cargado junto con las transformaciones necesarias para el preprocesamiento de imágenes y la tokenización de consultas textuales.

## Generación de embeddings

Todas las imágenes del dataset fueron procesadas mediante el encoder visual de CLIP para obtener una representación vectorial de 512 dimensiones. Los embeddings resultantes fueron normalizados utilizando norma L2 con el objetivo de trabajar posteriormente con similitud coseno.

Como resultado se obtuvo una matriz de embeddings de dimensión **(17125, 512)** donde cada fila representa una imagen y cada columna una característica del espacio semántico aprendido por CLIP.

Los embeddings fueron almacenados en disco para evitar recalcularlos en ejecuciones posteriores.

## Clustering exploratorio

En esta sección se realizó un análisis exploratorio sobre los embeddings globales de 512 dimensiones generados por CLIP. Seleccionamos una muestra aleatoria de 2000 imágenes y las agrupamos aplicando K-Means, en 20 clusters según su similitud semántica. 

Posteriormente, se aplicó t-SNE para proyectar los embeddings a dos dimensiones y facilitar así su visualización.

<!-- FOTO -->

El gráfico demuestra que los embeddings generados por el modelo CLIP son altamente efectivos para capturar las características semánticas y visuales de las imágenes. La existencia de múltiples grupos (clusters) claramente separados indica que el modelo logra diferenciar y clasificar el conjunto de 2000 imágenes en categorías conceptuales distintas. 

También se puede notar la existencia de clusters (0, 2, 5, 9 y 10) que están densamente empaquetados pero que a la vez se encuentran separados del resto; y por el contrario, tenemos clusters superpuestos en la región central con una mayor dispersión, lo que sugiere conceptos visuales más amplios o ambiguos que dificultan una separación estricta de las imágenes.

Para interpretar el contenido de cada cluster, se visualizaron ejemplos cercanos al centroide (representativos) y ejemplos ubicados en los bordes del cluster (casos límite).

<!-- FOTO -->

Con esta imágen podemos notar que el modelo logra agrupar las imágenes basándose en conceptos semánticos muy claros, más allá de simples similitudes de color o forma. Por ejemplo, para el cluster 0 la temática es caballos y equitación, mientras que para el cluster 2 la temática son gatos. Pero también debemos destacar que mientras en el centro del cluster el objeto principal está bien encuadrado y representa la idea central del grupo, al acercarnos a los límites o bordes del cluster podemos notar el impacto del fondo y contexto, por ejemplo, en el cluster 1 la temática principal son las motos, pero en las imágenes menos representativas encontramos una botella de aceite de motor y una camioneta.

## Construcción del índice FAISS

Para realizar búsquedas eficientes se utilizó FAISS (Facebook AI Similarity Search).

Se implementó un índice de tipo **IndexFlatIP**, que utiliza producto interno como métrica de similitud. Dado que los embeddings se encuentran normalizados, esta métrica equivale a la similitud coseno entre vectores.

El índice construido contiene 17.125 vectores indexados de 512 dimensiones cada uno.

## Búsqueda texto → imagen (baseline)

La estrategia baseline consiste en transformar una consulta textual en un embedding utilizando CLIP y posteriormente recuperar las imágenes más similares mediante FAISS.

El procedimiento implementado comprende la generación del embedding textual, su normalización y la búsqueda de los vecinos más cercanos dentro del índice. Como resultado, el sistema devuelve un ranking de imágenes ordenadas según su similitud con la consulta realizada.

Realizamos una búsqueda de imágenes para consultas simples en inglés (dog, car, person) y en español, y consultas con negaciones para identificar las limitaciones del baseline. Los resultados muestran que CLIP recupera adecuadamente conceptos positivos, pero presenta dificultades para interpretar restricciones negativas de forma directa.

<!-- TODO: Insertar ejemplos visuales de búsquedas baseline (dog, car, person) -->

# Capa agéntica: reformulación y validación de consultas

## Procesamiento de consultas mediante LLM

Con el objetivo de mejorar la interacción con el sistema, se incorporó una capa agéntica basada en un modelo de lenguaje ejecutado mediante OpenRouter. Para ello se utilizó el modelo Gemma 4 31B Instruct.

## Detección de idioma y traducción

Dado que CLIP fue entrenado principalmente en inglés, se implementó un agente encargado de detectar automáticamente el idioma de la consulta y traducirla cuando fuera necesario.

El agente devuelve el idioma detectado, la traducción al inglés y una indicación de si la consulta fue modificada. Esta etapa permite realizar búsquedas consistentes independientemente del idioma utilizado por el usuario.

<!-- TODO: Insertar ejemplos de traducción español → inglés -->

## Detección de negaciones

Se implementó un segundo agente encargado de identificar términos excluyentes dentro de la consulta.

El objetivo es separar la parte positiva de la búsqueda de los conceptos que deben excluirse. Por ejemplo, una consulta como "car not red" es transformada en una componente positiva ("car") y una lista de términos negativos ("red").

La información obtenida será utilizada posteriormente para mejorar la recuperación y el ordenamiento de resultados.

| Consulta                   | Positivo | Negativo    |
| -------------------------- | -------- | ----------- |
| car not red                | car      | red         |
| dog without collar         | dog      | collar      |
| bus not yellow and not old | bus      | yellow, old |

<!-- TODO: Insertar ejemplos adicionales de consultas con negaciones -->

## Uso de prompts estructurados

Para garantizar respuestas consistentes por parte del modelo de lenguaje, se utilizaron prompts con formatos de salida estrictamente definidos. Esto permitió simplificar el procesamiento posterior y reducir errores de interpretación durante la extracción de resultados.

# Reranking para negaciones y atributos excluyentes

Para mejorar los resultados en consultas que contienen negaciones, se implementó un esquema de reranking basado en penalización de similitud.
 
El procedimiento consiste en recuperar 50 candidatos usando únicamente la componente positiva de la consulta. Posteriormente, para cada candidato se calcula su similitud con los términos negativos detectados. El score final se obtiene como:
 
```
score_final = score_positivo − PESO_PENALIZACION × score_negativo
```

**Parámetro:** PESO_PENALIZACION = 0.6, elegido por experimentación visual. Un valor mayor produce penalizaciones más agresivas que pueden degradar la recuperación del objeto principal.

# Evaluación de resultados

## Métrica AP@10 / mAP@10

La métrica utilizada es **Average Precision @10 (AP@10)**, que considera tanto la relevancia como el orden de los resultados. Se implementó localmente para evaluar cada configuración antes de subir a Kaggle. El **mAP@10** promedia el AP@10 sobre todas las consultas del conjunto.

## Ground truth de q1..q20 (clases VOC)

Para las 20 clases de Pascal VOC, el ground truth corresponde al conjunto de imágenes que contienen esa clase según las anotaciones oficiales. Los resultados del baseline son:
 
 --- Insertar Tabla Actualizada ---

## Esquema propio para q21..q40 (consultas complejas)

Se diseñó un esquema de evaluación basado en intersección de resultados parciales. Cada consulta compleja se descompone en componentes simples en inglés, y el ground truth aproximado se construye como la intersección de los top-200 resultados de cada componente.
 
Ejemplo: `"persona montando un caballo en un campo abierto"` → componentes `["person riding horse", "horse field"]` → 44 imágenes relevantes.
 
Todas las consultas obtuvieron ground truth no vacío, con un rango de imágenes relevantes entre 31 y 168 por consulta.

## Comparación: baseline vs reformulación vs +reranking

--- Insertar Tabla y Gráficos Actualizados ---

## Ablation study

En esta sección aislamos el aporte de cada componente (CLIP solo, +LLM, +reranking) para mostrar cuánto suma cada uno.

--- Insertar Tabla y Gráficos Actualizados ---

# Generación de submission.csv (Kaggle)

El archivo submission.csv contiene una fila por consulta (q1 a q40) con las columnas qid y preds. Cada fila incluye exactamente 10 image IDs sin extensión .jpg, separados por ;, ordenados de mayor a menor relevancia.

Las consultas q1-q20 corresponden a las 20 clases VOC buscadas en inglés con el baseline. Las consultas q21-q40 corresponden a consultas complejas obtenidas desde solution.csv de la cátedra.

# Conclusiones

--- Establecer conclusiones finales del proyecto. Describir los problemas con los que nos encontramos (Por ejemplo, acá podríamos poner el problema que tuvimos con Reranking). Empezamos haciendo esto ... lo cual nos daba los siguientes resultados, por lo tanto, decidimos modificar esto y la mejora fue ...

Nos dimos cuenta que no era necesario realizar XXXX; etc ---
