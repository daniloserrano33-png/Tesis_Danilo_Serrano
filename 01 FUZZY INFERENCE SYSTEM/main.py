# src/model/main.py

from pathlib import Path

from src.preprocess import preprocessing as pp
from src.loader import data_loader as dl
from src.model import evaluate_model as em
from src.model import train_model as tm
import os
from os import path

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

if __name__ == "__main__":
    
    from config import LOAN_FILE, MODEL_DIR, PKL_DIR
    
    df = dl.load_data(LOAN_FILE)
    df = dl.inspect_and_prepare_data(df)
    df = df.drop(columns=['recoveries'], errors='ignore')
    dl.print_dataframe_stats(df)

    print("[INFO] Data loaded and preprocessed successfully.")

    df_processed, dropped_missing = pp.handle_missing_values(df.copy(), missing_percentage_threshold=30)
    df_processed = pp.convert_specific_categorical_to_numeric(df_processed)
    df_processed = pp.impute_missing_values(df_processed)
    df_processed, dropped_low_variance = pp.drop_low_variance_features(df_processed, variance_threshold=0.01)
    df_processed, dropped_descriptive = pp.drop_descriptive_features(df_processed)
    df_processed, dropped_correlated = pp.drop_highly_correlated_features(df_processed, correlation_threshold=0.6)
    df_processed, dropped_low_ks = pp.drop_low_ks_features(df_processed, target_col='bad_good', ks_threshold=0.03)
    df_processed, dropped_low_iv, iv_df = pp.drop_low_iv_categorical_features(df_processed, df_processed.select_dtypes(include=['object']).columns.tolist(), target='bad_good', iv_threshold=0.02)

    df_processed, dropped_low_ks_iv = tm.select_top_features_manual(df_processed, target='bad_good')

    print("\n[INFO] Processed DataFrame head:")
    print(df_processed.head())
    print("\n[INFO] Processed DataFrame info:")    
    df_processed.info()

    print(f"\n[INFO] Processed DataFrame shape: {df_processed.shape}")

    # Separar test final
    X_train, X_test, y_train, y_test = tm.split_data(df_processed, target_col='bad_good', test_size=0.3)

    # ── Paso 2: dividir el train en porción Base (70%) y porción COSI (30%) ────
    # Porción Base  → entrena los modelos de ensemble (CatBoost / RF / LightGBM).
    # Porción COSI  → datos "vírgenes" que el modelo base nunca vio.
    #                 RuleCOSI+ los usa para extraer reglas interpretables sin
    #                 que el modelo base haya memorizado esos ejemplos → sin fuga.
    X_train_base, X_train_cosi, y_train_base, y_train_cosi = tm.split_data_base_cosi(
        X_train, y_train, cosi_size=0.30
    )

    # Guardar todas las particiones como pickle
    tm.write_pkl_file(X_train,      path.join(PKL_DIR, "X_train.pkl"))           # train completo (referencia)
    tm.write_pkl_file(y_train,      path.join(PKL_DIR, "y_train.pkl"))
    tm.write_pkl_file(X_test,       path.join(PKL_DIR, "X_test.pkl"))            # test final
    tm.write_pkl_file(y_test,       path.join(PKL_DIR, "y_test.pkl"))
    tm.write_pkl_file(X_train_base, path.join(PKL_DIR, "X_train_base.pkl"))      # porción base
    tm.write_pkl_file(y_train_base, path.join(PKL_DIR, "y_train_base.pkl"))
    tm.write_pkl_file(X_train_cosi, path.join(PKL_DIR, "X_train_cosi.pkl"))      # porción COSI
    tm.write_pkl_file(y_train_cosi, path.join(PKL_DIR, "y_train_cosi.pkl"))

    print("\n[INFO] Data split into training and testing sets and saved as pickle files.")
    print(f"       X_train_base : {X_train_base.shape}  (entrena los modelos base)")
    print(f"       X_train_cosi : {X_train_cosi.shape}  (exclusivo para RuleCOSI+)")
    print(f"       X_test       : {X_test.shape}        (evaluación final)")

    # ── Paso 3: entrenar los modelos BASE solo con X_train_base ─────────────────
    rf_pipeline, rf_metrics, X_train_encoded, cat_mappings = tm.train_random_forest_optimized(
        X_train_base, y_train_base,
        n_estimators=200,
        max_features='sqrt',
        max_depth=5,
        verbose=True
    )

    rf_paths = tm.save_model(
        model=rf_pipeline,
        metrics=rf_metrics,
        X_train_encoded=X_train_encoded,
        category_mappings=cat_mappings,
        model_name="random_forest_DANI_encoded",
        output_dir=MODEL_DIR
    )

    catboost_model, catboost_metrics = None, None

    catboost_model, catboost_metrics, X_train_encoded, cat_mappings = tm.train_catboost_optimized(
        X_train_base, y_train_base,
        depth=5,
        iterations=200,
        verbose=True,
        learning_rate=1
        )

    cb_paths = tm.save_model(
        model=catboost_model,
        metrics=catboost_metrics,
        X_train_encoded=X_train_encoded,
        category_mappings=cat_mappings,
        model_name="catboost_DANI_encoded",
        output_dir=MODEL_DIR
    )

    lightgbm_model, lightgbm_metrics = None, None

    lightgbm_model, lightgbm_metrics, X_train_encoded, cat_mappings = tm.train_lightgbm_optimized(
        X_train_base, y_train_base,
        max_depth=5,
        n_estimators=200,
        validation_split=0.3,
        verbose=True
    )
    
    lightgbm_paths = tm.save_model(
        model=lightgbm_model,
        metrics=lightgbm_metrics,
        X_train_encoded=X_train_encoded,
        category_mappings=cat_mappings,
        model_name="lightgbm_DANI_encoded",
        output_dir=MODEL_DIR
    )
    
    comparison = tm.compare_models(
        rf_metrics=rf_metrics,
        catboost_metrics=catboost_metrics,
        lightgbm_metrics=lightgbm_metrics,
        focus_metric='Train F1'
    )

    
    print("\n[INFO] -------------------- Models Saved --------------------")

    print(f"\n[INFO] Random Forest:")
    print(f"  {rf_paths['model']}")
    print(f"\n[INFO] CatBoost:")
    print(f"  {cb_paths['model']}")
    print(f"\n[INFO] LightGBM:")
    print(f"  {lightgbm_paths['model']}")
    print("\n[INFO] -------------------- Save Complete --------------------")

    print("\n[INFO] CatBoost model evaluating start.")

    X_test_encoded = tm.encode_data_like_training(X_test, cat_mappings)
    threshold = catboost_metrics['optimal_threshold']
    em.evaluate_model(catboost_model, X_test_encoded, y_test, threshold=threshold)
    print("\n[INFO] CatBoost model evaluating successfully.")
    print(cat_mappings)

    print("For analizing fuzzy inference system go to Execute Fuzzy Inference.ipynb")

    # ══════════════════════════════════════════════════════════════════════════
    # ── RuleCOSI+: modelo seleccionado por frontera de Pareto ─────────────────
    #    Configuración: α=0.50, β=0.01, c=0.10  →  5 reglas  (Escenario B)
    # ══════════════════════════════════════════════════════════════════════════
    import sys as _sys
    import pickle as _pk
    from sklearn.metrics import f1_score as _f1_fn, precision_score as _prec_fn, recall_score as _rec_fn

    _project_dir = path.dirname(path.abspath(__file__))
    _vendor_path = path.join(_project_dir, 'vendor', 'rulecosi')
    if _vendor_path not in _sys.path:
        _sys.path.insert(0, _vendor_path)

    from rulecosi import RuleCOSIClassifier

    print("\n" + "=" * 62)
    print("[INFO] RuleCOSI+  —  Pareto-óptima (codo de Pareto, punto P2)")
    print("=" * 62)

    _ALPHA = 0.50   # conf_threshold
    _BETA  = 0.01   # cov_threshold
    _C     = 0.10   # poda C4.5

    print(f"  conf_threshold (α) : {_ALPHA}")
    print(f"  cov_threshold  (β) : {_BETA}")
    print(f"  c  (poda C4.5)     : {_C}")

    if catboost_model is not None:
        # Cargar partición COSI (sin recoveries, 30 % de X_train)
        with open(path.join(PKL_DIR, 'X_train_cosi.pkl'), 'rb') as _fh:
            _X_cosi_raw = _pk.load(_fh)
        with open(path.join(PKL_DIR, 'y_train_cosi.pkl'), 'rb') as _fh:
            _y_cosi = _pk.load(_fh)

        _X_cosi_enc = tm.encode_data_like_training(_X_cosi_raw, cat_mappings)
        print(f"\n[INFO] Partición COSI  : {_X_cosi_enc.shape[0]:,} muestras × {_X_cosi_enc.shape[1]} features")

        print("[INFO] Entrenando RuleCOSI+ (puede tardar unos minutos) ...")
        _cosi_model = RuleCOSIClassifier(
            base_ensemble=catboost_model,
            conf_threshold=_ALPHA,
            cov_threshold=_BETA,
            c=_C,
            random_state=42
        )
        _cosi_model.fit(_X_cosi_enc, _y_cosi)

        # Estadísticas del ruleset
        _all_rules    = _cosi_model.simplified_ruleset_.rules
        _active_rules = [r for r in _all_rules if len(r.A) > 0]
        _n_rules      = len(_active_rules)
        _avg_conds    = sum(len(r.A) for r in _active_rules) / _n_rules if _n_rules else 0
        _n_baseline   = sum(len(rs.rules) for rs in _cosi_model.original_rulesets_)
        _redu         = round(1 - _n_rules / _n_baseline, 4) if _n_baseline > 0 else 0.0

        print(f"\n[INFO] Reglas activas      : {_n_rules}")
        print(f"[INFO] Reglas baseline     : {_n_baseline:,}")
        print(f"[INFO] Condiciones/regla   : {_avg_conds:.2f}")
        print(f"[INFO] REDU                : {_redu:.4f}")


        print('\n=== Métricas de las Reglas del modelo seleccionado ===')
        for i, rule in enumerate(_all_rules):
            print(f'Regla {i+1}: {rule}')


        print("\n[INFO] Detalle de reglas extraídas:")
        for _i, _r in enumerate(_all_rules):
            _tag = "  [DEFAULT]" if len(_r.A) == 0 else ""
            print(f"  Regla {_i}{_tag}  →  clase={_r.class_index}  conf={_r.conf:.4f}  cov={_r.cov:.4f}")
            for _cond in _r.A:
                print(f"    {_cond}")

        # Evaluación sobre X_test
        print(f"\n[INFO] Evaluando sobre X_test ({X_test_encoded.shape[0]:,} muestras) ...")
        _y_pred_cosi = _cosi_model.predict(X_test_encoded)
        _f1_cosi    = _f1_fn(y_test,   _y_pred_cosi, zero_division=0)
        _prec_cosi  = _prec_fn(y_test, _y_pred_cosi, zero_division=0)
        _rec_cosi   = _rec_fn(y_test,  _y_pred_cosi, zero_division=0)

        # Métricas CatBoost sobre el mismo X_test (umbral óptimo)
        _thresh_cb   = catboost_metrics['optimal_threshold']
        _proba_cb    = catboost_model.predict_proba(X_test_encoded)[:, 1]
        _y_pred_cb   = (_proba_cb >= _thresh_cb).astype(int)
        _f1_cb       = _f1_fn(y_test,   _y_pred_cb, zero_division=0)
        _prec_cb     = _prec_fn(y_test, _y_pred_cb, zero_division=0)
        _rec_cb      = _rec_fn(y_test,  _y_pred_cb, zero_division=0)

        print("\n" + "=" * 62)
        print("[INFO] COMPARATIVA FINAL — CatBoost vs RuleCOSI+")
        print("=" * 62)
        print(f"{'Métrica':<24} {'CatBoost':>10} {'RuleCOSI+':>11} {'Δ':>9}")
        print("-" * 56)
        print(f"{'F1-score':<24} {_f1_cb:>10.4f} {_f1_cosi:>11.4f} {(_f1_cosi - _f1_cb):>+9.4f}")
        print(f"{'Precision':<24} {_prec_cb:>10.4f} {_prec_cosi:>11.4f} {(_prec_cosi - _prec_cb):>+9.4f}")
        print(f"{'Recall':<24} {_rec_cb:>10.4f} {_rec_cosi:>11.4f} {(_rec_cosi - _rec_cb):>+9.4f}")
        print(f"{'Reglas activas':<24} {_n_baseline:>10,} {_n_rules:>11,}")
        print(f"{'REDU':<24} {'0.0000':>10} {_redu:>11.4f}")
        print(f"{'% F1 retenido':<24} {'100.00 %':>10} {(_f1_cosi / _f1_cb * 100 if _f1_cb > 0 else 0):>10.2f} %")
        print("=" * 62)
    else:
        print("[WARN] catboost_model no disponible; bloque RuleCOSI+ omitido.")