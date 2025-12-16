# DQM ML Pipeline - Testing & Usage Guide

Ce document contient les commandes pour tester et exécuter le pipeline DQM ML complet.

## 📋 Prérequis

Assurez-vous d'avoir activé l'environnement virtuel :

```bash
source .venv/bin/activate
```

## 🔧 Pipeline Complet

### Étape 1 : Extraction des Features Visuelles

Calcule les métriques visuelles (luminosité, contraste, blur, entropie) à partir des images.

```bash
dqm-ml process -p packages/dqm-ml-pipeline/config/visual_features.yaml
```

**Entrée :**
- Fichier source : `/Users/checkkoutame/Desktop/Renare/kc-data-sampling/raw_samples/samples_raw_with_inference.parquet`
  - ou tout autre parquet
- Colonnes requises : `image_bytes`, métadonnées des échantillons

**Sortie :**
- Fichier : `features/samples_raw_with_inference.parquet`
- Nouvelles colonnes : `m_luminosity`, `m_contrast`, `m_blur_level`, `m_entropy`

### Étape 2 : Calcul des Métriques de Représentativité

Calcule les métriques de représentativité (Chi-Square, KS, Shannon Entropy, GRTE) sur les features.

#### 2.1 Métriques par Split (train/val/test)

```bash
python scripts/duckdb_subsets.py \
  -i features/samples_raw_with_inference.parquet \
  -c packages/dqm-ml-pipeline/config/representativness.yaml \
  -g split \
  -o metrics_outputs/representativness/subsets_splits/all_metrics.parquet \
  -j metrics_outputs/representativness/subsets_splits/data_with_metrics.parquet
```

**Paramètres :**
- `-i` : Fichier d'entrée avec les features visuelles
- `-c` : Configuration des métriques de représentativité
- `-g` : Groupement par colonne (`split`)
- `-o` : Sortie agrégée (métriques résumées)
- `-j` : Sortie jointe (données + métriques par échantillon)

**Sorties :**
- `all_metrics.parquet` : Métriques agrégées par split
- `data_with_metrics.parquet` : Données originales + colonnes de métriques

#### 2.2 Métriques par Split ET Class

```bash
python scripts/duckdb_subsets.py \
  -i features/samples_raw_with_inference.parquet \
  -c packages/dqm-ml-pipeline/config/representativness.yaml \
  -g split class_name \
  -o metrics_outputs/representativness/subsets_splits_class_name/all_metrics.parquet \
  -j metrics_outputs/representativness/subsets_splits_class_name/data_with_metrics.parquet
```

**Groupement double :** Métriques calculées pour chaque combinaison `(split, class_name)`
