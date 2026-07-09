# Extracción de Reglas Interpretables para Scoring Crediticio mediante RuleCOSI+

**Tesis de Maestría — Universidad San Francisco de Quito**
**Danilo Fernando Serrano Enríquez · Maestría en Ciencia de Datos**

---

## Objetivo

Extraer un conjunto mínimo de reglas crisp interpretables a partir de un modelo CatBoost entrenado sobre el dataset de LendingClub, utilizando el algoritmo **RuleCOSI+** (Obregón & Jung, 2023) y seleccionando la configuración óptima mediante análisis de **frontera de Pareto**. El resultado final es un modelo de **5 reglas** que retiene el **98,9 % del F1-score** del CatBoost base con una reducción de reglas **REDU = 0,9849** (de 332 reglas activas a 5).

Un aporte metodológico adicional es la identificación y corrección de una **fuga de información** (data leakage) presente en el trabajo previo de Jaramillo et al. (2024): la variable `recoveries` es post-hoc y se elimina antes de cualquier entrenamiento.

---

## Resultados Principales

| Métrica | Valor |
|---|---|
| CatBoost F1 (test, sin recoveries) | 0,7740 |
| RuleCOSI+ F1 (test) | **0,7652** |
| % del CatBoost retenido | **98,9 %** |
| Número de reglas | **5** |
| REDU | **0,9849** |
| Parámetros óptimos | α=0,50 · β=0,01 · c=0,10 |

**Frontera de Pareto (3 puntos Pareto-óptimos):**

| Punto | Reglas | F1 | Descripción |
|---|---|---|---|
| P1 (codo) | 2 | 0,7591 | Máxima compresión |
| **P2 ★** | **5** | **0,7652** | **Seleccionado — mejor equilibrio** |
| P3 | 23 | 0,7671 | Rendimiento máximo |

---

## Estructura del Repositorio

```
Tesis Danilo/
├── README.md
├── .gitignore
├── tesis_danilo_ieee.tex             ← Documento de tesis en formato LaTeX (IEEEtran)
├── tesis_danilo_overleaf.zip         ← Paquete listo para subir a Overleaf
│
├── docs/
│   ├── Revision 08_07_2026.docx     ← Versión revisada del documento de tesis
│   ├── Revision 08_07_2026.pdf      ← Versión PDF de la revisión
│   ├── tesis_danilo_v2.docx          ← Versión anterior del documento
│   ├── obregon2023.pdf               ← Paper original de RuleCOSI+
│   ├── bitacora_proyecto_rulecosi.docx
│   └── introduction.docx
│
└── 01 FUZZY INFERENCE SYSTEM/        ← Código fuente principal
    ├── main.py                       ← Pipeline completo (entry point)
    ├── config.py                     ← Rutas y constantes
    │
    ├── data/
    │   └── LCDataDictionary.xlsx     ← Diccionario de variables
    │   # Loan_status_2007-2020Q3.gzip no se sube a git (>1 GB)
    │
    ├── src/
    │   ├── loader/data_loader.py     ← Carga y preparación de datos
    │   ├── preprocess/preprocessing.py  ← Limpieza y selección de features
    │   ├── model/train_model.py      ← Entrenamiento CatBoost / RF / LightGBM
    │   ├── model/evaluate_model.py   ← Métricas y evaluación
    │   ├── fuzzy/fuzzy_inference.py  ← Motor FIS (referencia)
    │   └── rules/extract_crisp_rules.py ← Extracción de reglas crisp
    │
    ├── notebooks/
    │   ├── Execute Fuzzy Inference.ipynb
    │   └── experiments/
    │       ├── cosi_experimental_grid_v3.ipynb  ← EXPERIMENTO PRINCIPAL
    │       │   # Grid 24 configs · Pareto · 5 reglas finales (sin recoveries)
    │       ├── barras_nreglas_v3.png    ← Figura 1: reducción de reglas
    │       ├── heatmap_metricas_v3.png  ← Figura 2: heatmap del grid
    │       └── pareto_cosi_v3.png       ← Figura 3: frontera de Pareto
    │
    ├── object/
    │   ├── model/   ← modelos entrenados (.cbm, .txt, .pkl, .parquet)
    │   └── pkl/     ← particiones serializadas (X_train_cosi, X_test, etc.)
    │   # Ambas carpetas están en .gitignore por tamaño
    │
    └── vendor/rulecosi/              ← RuleCOSI+ (vendoreado, sin modificar)
```

---

## Cómo Reproducir

### 1. Requisitos

```bash
pip install catboost lightgbm scikit-learn pandas numpy matplotlib seaborn
```

RuleCOSI+ está incluido como dependencia vendoreada en `vendor/rulecosi/`. No requiere instalación adicional.

### 2. Datos

Descargar el dataset de LendingClub desde [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) y ubicarlo en:

```
01 FUZZY INFERENCE SYSTEM/data/Loan_status_2007-2020Q3.gzip
```

### 3. Pipeline completo

```bash
cd "01 FUZZY INFERENCE SYSTEM"
python main.py
```

Esto ejecuta en orden: carga → preprocesamiento → eliminación de `recoveries` → partición anti-leakage → entrenamiento (Random Forest, CatBoost, LightGBM) → evaluación → extracción de reglas RuleCOSI+ con la configuración seleccionada (α=0,50 · β=0,01 · c=0,10).

### 4. Experimento completo (grid de 24 configuraciones + Pareto)

Abrir y ejecutar el notebook:

```
01 FUZZY INFERENCE SYSTEM/notebooks/experiments/cosi_experimental_grid_v3.ipynb
```

Este notebook genera las tres figuras del documento de tesis y la tabla de resultados del grid.

---

## Comparación con el Trabajo de Byron (Jaramillo et al., 2024)

| Enfoque | Reglas | F1 | % del CB propio | recoveries |
|---|---|---|---|---|
| CatBoost Byron (con leakage) | — | 0,8887 | 100 % | Sí |
| FIS Mamdani — Byron | 717 | 0,4963 | 55,8 % | Sí |
| CatBoost limpio (este trabajo) | — | 0,7740 | 100 % | No |
| **RuleCOSI+ P2 — este trabajo** | **5** | **0,7652** | **98,9 %** | **No** |

La diferencia de F1 entre el CatBoost de Byron (0,8887) y el de este trabajo (0,7740) cuantifica el componente espurio aportado por `recoveries`: **0,1147 puntos de inflación artificial**.

---

## Referencia

Obregón, L. & Jung, A. (2023). *RuleCOSI+: Rule Extraction Algorithm for Accurate and Interpretable Classification*. Expert Systems with Applications, 216, 119432. https://doi.org/10.1016/j.eswa.2022.119432
