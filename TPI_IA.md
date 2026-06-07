# Trabajo Práctico Integrador - Otoño 2026

**Materia:** Inteligencia Artificial
**Cuatrimestre:** 1° Cuatrimestre
**Año:** 2026
**Grupo:** *Completar*

## Integrantes

* Figueroa Prada, Juan Uriel – 407213
* Gomez Alameda, Romina Abigail – 85296
* Saggiorato, Gina – 95794
* Storello Chiofalo, Juan Ignacio – 85408

# Análisis del Dataset (EDA)

## Estadísticas básicas

Se analizaron las imágenes del dataset Pascal VOC 2012. El conjunto contiene **17.125 imágenes**.

Sobre una muestra aleatoria de 500 imágenes se obtuvieron las siguientes estadísticas:

| Métrica         | Ancho (px) | Alto (px) | Tamaño (KB) |
| --------------- | ---------- | --------- | ----------- |
| Promedio        | 464.3      | 391.2     | 109.1       |
| Desvío estándar | 63.2       | 66.0      | 42.5        |
| Mínimo          | 200        | 111       | 16.0        |
| Máximo          | 500        | 500       | 242.7       |

## Distribución por clases

Las anotaciones fueron utilizadas para obtener estadísticas descriptivas. Cada clase fue contabilizada una única vez por imagen.

Las clases más frecuentes fueron:

| Clase  | Imágenes |
| ------ | -------: |
| person |     9583 |
| chair  |     1366 |
| dog    |     1341 |
| car    |     1284 |
| cat    |     1128 |

Las clases menos representadas fueron cow (340), sheep (357) y bus (467).

## Visualizaciones

Se generaron:

* Scatter de dimensiones de imagen.
* Histograma de tamaños de archivo.
* Distribución de imágenes por clase.
* Muestra aleatoria de imágenes del dataset.

# Metodología

## Baseline: CLIP + FAISS

Se utilizó CLIP ViT-B/32 para generar embeddings visuales de 512 dimensiones para las 17.125 imágenes del dataset.

Los embeddings fueron normalizados y almacenados en una matriz de dimensión **(17125, 512)**.

Para la recuperación se construyó un índice FAISS de tipo **IndexFlatIP**, utilizando similitud coseno sobre los embeddings normalizados.

## Búsqueda texto → imagen

Las consultas se transforman en embeddings mediante CLIP y se recuperan los k vecinos más cercanos utilizando FAISS.

El sistema devuelve las imágenes ordenadas según su similitud con la consulta.

## Procesamiento de consultas mediante LLM

Se incorporó un modelo Gemma 4 31B Instruct mediante OpenRouter para asistir el procesamiento de consultas.

### Traducción automática

Las consultas son analizadas para detectar su idioma y, cuando corresponde, traducidas al inglés antes de realizar la búsqueda.

### Detección de negaciones

Se implementó un agente que identifica componentes positivos y negativos de la consulta.

Ejemplos:

| Consulta                   | Positivo | Negativo    |
| -------------------------- | -------- | ----------- |
| car not red                | car      | red         |
| dog without collar         | dog      | collar      |
| bus not yellow and not old | bus      | yellow, old |

### Prompts estructurados

Los agentes utilizan formatos de salida controlados para facilitar el procesamiento automático de las respuestas.

---

**FALTA:**Resultados, Discusión y Trabajo Futuro 
