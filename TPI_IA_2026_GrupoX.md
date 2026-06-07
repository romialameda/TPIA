# Trabajo Práctico Integrador

## Materia y Cátedra
**Materia:** Inteligencia Artificial  
**Cátedra:** Inteligencia Artificial

## Cuatrimestre y Año
**1.º Cuatrimestre 2026**

## Grupo
**Grupo:** A definir

## Integrantes
- Figueroa Prada, Juan Uriel – 407213
- Gomez Alameda, Romina Abigail – 85296
- Saggiorato, Gina – 95794
- Storello Chiofalo, Juan Ignacio – 85408

## # Análisis del Dataset (EDA)

## Estadísticas básicas

El dataset utilizado corresponde a Pascal VOC 2012 y contiene un total de **17.125 imágenes**. Para realizar el análisis exploratorio se tomó una muestra aleatoria de 500 imágenes, sobre la cual se calcularon estadísticas descriptivas relacionadas con dimensiones y tamaño de archivo.

Las imágenes presentan un ancho promedio de **464,3 píxeles** y una altura promedio de **391,2 píxeles**. El tamaño promedio de los archivos es de **109,1 KB**, con dimensiones mínimas de **200 × 111 píxeles** y máximas de **500 × 500 píxeles**.

| Métrica         | Ancho (px) | Alto (px) | Tamaño (KB) |
| --------------- | ---------- | --------- | ----------- |
| Promedio        | 464.3      | 391.2     | 109.1       |
| Desvío estándar | 63.2       | 66.0      | 42.5        |
| Mínimo          | 200        | 111       | 16.0        |
| Máximo          | 500        | 500       | 242.7       |

## Distribución por clases

Las anotaciones provistas por Pascal VOC 2012 fueron utilizadas para obtener estadísticas descriptivas sobre las clases presentes en el conjunto de datos. Para optimizar el tiempo de procesamiento, cada clase se contabilizó una única vez por imagen, independientemente de la cantidad de objetos pertenecientes a dicha clase presentes en la escena.

La clase más frecuente es **person**, presente en 9.583 imágenes. Le siguen **chair** (1.366), **dog** (1.341), **car** (1.284) y **cat** (1.128). Las clases menos representadas son **cow** (340), **sheep** (357) y **bus** (467).

Estos resultados evidencian un importante desbalance entre las distintas categorías del dataset, especialmente debido a la alta presencia de imágenes que contienen personas.

## Visualizaciones

Se generaron distintas visualizaciones con el objetivo de comprender mejor las características del conjunto de datos. En particular, se analizaron las dimensiones de las imágenes mediante un gráfico de dispersión, la distribución de tamaños de archivo mediante histogramas y la cantidad de imágenes por clase mediante gráficos de barras.

Adicionalmente, se seleccionó una muestra aleatoria de imágenes representativas para realizar una inspección visual del contenido del dataset y verificar la diversidad de escenas y objetos presentes.

## Clustering exploratorio

Como análisis exploratorio complementario, se realizó un agrupamiento de imágenes utilizando embeddings globales obtenidos mediante CLIP. Posteriormente, estos vectores fueron proyectados a un espacio de menor dimensión para facilitar su visualización y análisis.

El objetivo de este proceso fue identificar agrupaciones naturales dentro del dataset y evaluar si imágenes con contenido visual similar tendían a ubicarse próximas entre sí en el espacio semántico generado por los embeddings. Los resultados obtenidos se presentan junto con la generación de embeddings y el análisis de similitud semántica.


