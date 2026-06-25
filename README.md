# Desarrollo y Evaluación de Modelos de Aprendizaje Automático
### Predicción de precios inmobiliarios en el mercado español

Trabajo de Fin de Grado — Grado en Informática (GIF), Universidad Alfonso X el Sabio.

---

## Descripción

Este proyecto desarrolla y evalúa un *pipeline* completo de aprendizaje automático para la estimación del precio de viviendas en España, comparando cinco familias de modelos de regresión y analizando la interpretabilidad del mejor modelo mediante SHAP.

**Pregunta de investigación:** ¿En qué medida los métodos de *gradient boosting* superan a redes neuronales y modelos lineales en la predicción de precios inmobiliarios en un mercado nacionalmente heterogéneo como el español, y qué variables —físicas, geoeconómicas y de ingeniería— determinan dichas predicciones?

##  Dataset

[Spanish Housing Dataset](https://www.kaggle.com/datasets/thedevastator/spanish-housing-dataset-location-size-price-and) (Kaggle): 100.000 anuncios inmobiliarios extraídos de Idealista (2019), enriquecidos con indicadores socioeconómicos provinciales del INE.

##  Modelos implementados

| Modelo | RMSE (€) | MAE (€) | R² |
|---|---|---|---|
| Regresión Lineal | 4,77 × 10¹³ | 7,21 × 10¹¹ | < 0 |
| Ridge | 214.098 | 125.800 | 0,5410 |
| Random Forest | 166.399 | 83.102 | 0,7227 |
| **XGBoost** | **163.610** | **83.442** | **0,7320** |
| MLP (red neuronal) | 173.655 | 91.635 | 0,6980 |

**Mejor modelo: XGBoost**, con interpretabilidad analizada mediante valores SHAP (`TreeExplainer`).

##  Pipeline

1. **EDA** — distribución del precio, dispersión precio-m², heterogeneidad por tipo de vivienda y zona
2. **Preprocesamiento** — imputación de valores ausentes, eliminación de outliers por percentil, *feature engineering* (`ratio_util_real`, `amenity_score`, `rooms_per_m2`), codificación OHE y estandarización Z-score
3. **Entrenamiento** — 5 modelos con `train_test_split` (80/20) y `RandomizedSearchCV` para Random Forest y XGBoost
4. **Evaluación** — RMSE, MAE, R² sobre el conjunto de test
5. **Interpretabilidad** — análisis SHAP sobre el mejor modelo (XGBoost)

##  Estructura del repositorio

```
.
├── TFG_Pipeline.ipynb       # Notebook completo del pipeline
├── data/
│   └── spanish_houses.csv   # Dataset (no incluido, ver Kaggle)
├── images/                  # Gráficas exportadas para la memoria
└── README.md
```

## Requisitos

- Python 3.12
- Ver `requirements.txt`

```bash
pip install -r requirements.txt
```

Dependencias principales: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `shap`, `matplotlib`, `seaborn`.

##  Ejecución

1. Descarga el dataset desde [Kaggle](https://www.kaggle.com/datasets/thedevastator/spanish-housing-dataset-location-size-price-and) y colócalo en `data/spanish_houses.csv`
2. Abre `TFG_Pipeline.ipynb` en Jupyter Notebook o VS Code
3. Ejecuta todas las celdas en orden (`Run All`)

> **Nota sobre reproducibilidad:** los modelos lineales, Random Forest y XGBoost usan `random_state=42` y son completamente reproducibles. El MLP (red neuronal) puede presentar pequeñas variaciones entre ejecuciones por la inicialización aleatoria de TensorFlow/Keras, salvo que se fije `tf.random.set_seed(42)` antes de su entrenamiento.

## 📈 Resultados clave (SHAP)

Las variables más influyentes en la predicción de XGBoost son, por orden: `bath_num` (número de baños), `m2_real` (superficie construida), y los indicadores geoeconómicos provinciales `population_prov` y `renta_media_prov`. Esto confirma que la ubicación geoeconómica explica el precio de un inmueble tanto o más que muchas de sus características físicas.
