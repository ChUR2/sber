from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from features import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_ACTIONS,
                      clean_sessions, make_features)

MODEL_VERSION = '1.0'


# Загрузка данных

def load_sessions(data_dir: Path) -> pd.DataFrame:
    csv, pkl = data_dir / 'ga_sessions.csv', data_dir / 'ga_sessions.pkl'
    if csv.exists():
        return pd.read_csv(csv, dtype=str)
    return pd.read_pickle(pkl).astype(str)


def load_target_sessions(data_dir: Path) -> set:
    """ID визитов, в которых было хотя бы одно целевое действие.

    ga_hits.csv весит около 4 ГБ, поэтому читаем его потоково.
    """
    csv, pkl = data_dir / 'ga_hits.csv', data_dir / 'ga_hits.pkl'
    positives: set = set()

    if csv.exists():
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            import pyarrow.csv as pv

            reader = pv.open_csv(
                csv,
                read_options=pv.ReadOptions(block_size=32 * 1024 * 1024),
                convert_options=pv.ConvertOptions(
                    include_columns=['session_id', 'event_action'],
                    column_types={'session_id': pa.string(), 'event_action': pa.string()},
                ),
            )
            targets = pa.array(TARGET_ACTIONS)
            for batch in reader:
                mask = pc.is_in(batch.column('event_action'), value_set=targets)
                positives.update(pc.filter(batch.column('session_id'), mask).to_pylist())
            return positives
        except ImportError:
            pass

        for chunk in pd.read_csv(csv, usecols=['session_id', 'event_action'],
                                 dtype=str, chunksize=2_000_000):
            hit = chunk[chunk['event_action'].isin(TARGET_ACTIONS)]
            positives.update(hit['session_id'].tolist())
        return positives

    hits = pd.read_pickle(pkl)
    return set(hits.loc[hits['event_action'].isin(TARGET_ACTIONS), 'session_id'].astype(str))


def build_dataset(data_dir: Path):
    sessions = clean_sessions(load_sessions(data_dir))
    positives = load_target_sessions(data_dir)
    y = sessions['session_id'].isin(positives).astype('int8')
    return sessions, y


# Пайплайн

def build_pipeline() -> Pipeline:
    import lightgbm as lgb

    encoder = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value',
                                   unknown_value=-1,
                                   encoded_missing_value=-1,
                                   dtype=np.float64), CATEGORICAL_FEATURES),
            ('num', 'passthrough', NUMERIC_FEATURES),
        ],
        remainder='drop',
    )

    classifier = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        categorical_feature=list(range(len(CATEGORICAL_FEATURES))),
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    return Pipeline([
        ('features', FunctionTransformer(make_features, validate=False)),
        ('encoder', encoder),
        ('classifier', classifier),
    ])


def pick_threshold(y_true, proba) -> float:
    """Порог, максимизирующий F1 по положительному классу."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[int(np.nanargmax(f1[:-1]))])


# Основной сценарий

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--out', default='./model/sber_auto_model.pkl')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print('1. Читаю данные')
    t0 = time.time()
    sessions, y = build_dataset(data_dir)
    print(f'   визитов: {len(sessions):,}, доля целевых: {y.mean():.2%}, '
          f'{time.time() - t0:.0f} c')

    X_train, X_test, y_train, y_test = train_test_split(
        sessions, y, test_size=0.25, random_state=42, stratify=y)

    print('2. Бейзлайн (константа по частоте класса)')
    dummy = DummyClassifier(strategy='stratified', random_state=42).fit(X_train, y_train)
    auc_dummy = roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1])
    print(f'   ROC-AUC бейзлайна: {auc_dummy:.4f}')

    print('3. Обучаю LightGBM')
    t0 = time.time()
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    print(f'   обучение заняло {time.time() - t0:.0f} c')

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    print(f'   ROC-AUC: {auc:.4f}   PR-AUC: {ap:.4f}')

    # Классы сильно несбалансированы (2.7 процента), порог 0.5 почти всегда даёт 0.
    # Подбираем порог по максимуму F1, именно он попадает в API.
    threshold = pick_threshold(y_test, proba)
    print(f'   рабочий порог: {threshold:.4f}')
    print(classification_report(y_test, (proba >= threshold).astype(int), digits=3))

    metadata = {
        'name': 'SberAutopodpiska target action prediction',
        'author': 'Проектный практикум',
        'version': MODEL_VERSION,
        'date': datetime.now().isoformat(timespec='seconds'),
        'type': type(pipe.named_steps['classifier']).__name__,
        'roc_auc': float(auc),
        'pr_auc': float(ap),
        'baseline_roc_auc': float(auc_dummy),
        'target_actions': TARGET_ACTIONS,
        'threshold': float(threshold),
        'n_train': int(len(X_train)),
        'target_rate': float(y.mean()),
    }

    joblib.dump({'model': pipe, 'metadata': metadata}, out_path, compress=3)
    print(f'4. Модель сохранена: {out_path}')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
