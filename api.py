from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv('MODEL_PATH', Path(__file__).parent / 'model' / 'sber_auto_model.pkl'))

app = FastAPI(
    title='СберАвтоподписка: предсказание целевого действия',
    description='Принимает данные визита и возвращает вероятность целевого действия',
    version='1.0',
)

_bundle = joblib.load(MODEL_PATH)
model = _bundle['model']
metadata = _bundle['metadata']
THRESHOLD = float(metadata.get('threshold', 0.5))


# Схемы


class Visit(BaseModel):
    """Один визит на сайт. Все поля, кроме даты и времени, опциональны:
    пропуск обрабатывается так же, как отсутствующее значение в обучающих данных."""

    session_id: Optional[str] = None
    client_id: Optional[str] = None
    visit_date: str = Field(..., examples=['2021-12-28'])
    visit_time: str = Field(..., examples=['14:36:32'])
    visit_number: Optional[int] = 1
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_adcontent: Optional[str] = None
    utm_keyword: Optional[str] = None
    device_category: Optional[str] = None
    device_os: Optional[str] = None
    device_brand: Optional[str] = None
    device_model: Optional[str] = None
    device_screen_resolution: Optional[str] = None
    device_browser: Optional[str] = None
    geo_country: Optional[str] = None
    geo_city: Optional[str] = None

    model_config = {
        'json_schema_extra': {
            'example': {
                'session_id': '9055434745589932991.1637753792.1637753792',
                'client_id': '2108382700.1637753791',
                'visit_date': '2021-11-24',
                'visit_time': '14:36:32',
                'visit_number': 1,
                'utm_source': 'ZpYIoDJMcFzVoPFsHGJL',
                'utm_medium': 'banner',
                'utm_campaign': 'LEoPHuyFvzoNfnzGgfcd',
                'utm_adcontent': 'vCIpmpaGBnIQhyYNkXqp',
                'utm_keyword': 'puhZPIYqKXeFPaUviSjo',
                'device_category': 'mobile',
                'device_os': 'Android',
                'device_brand': 'Huawei',
                'device_screen_resolution': '360x720',
                'device_browser': 'Chrome',
                'geo_country': 'Russia',
                'geo_city': 'Zlatoust',
            }
        }
    }


class Prediction(BaseModel):
    session_id: Optional[str]
    prediction: int = Field(..., description='1 если ожидается целевое действие, иначе 0')
    probability: float = Field(..., description='Вероятность целевого действия')
    threshold: float


class BatchRequest(BaseModel):
    visits: List[Visit]

# Эндпойнты


@app.get('/status')
def status() -> dict:
    return {'status': 'ok', 'model_loaded': True}


@app.get('/version')
def version() -> dict:
    return metadata


@app.post('/predict', response_model=Prediction)
def predict(visit: Visit) -> Prediction:
    return _predict_frame(pd.DataFrame([visit.model_dump()]))[0]


@app.post('/predict_batch', response_model=List[Prediction])
def predict_batch(request: BatchRequest) -> List[Prediction]:
    if not request.visits:
        raise HTTPException(status_code=400, detail='Список визитов пуст')
    if len(request.visits) > 10_000:
        raise HTTPException(status_code=413, detail='Не более 10000 визитов за запрос')
    return _predict_frame(pd.DataFrame([v.model_dump() for v in request.visits]))


def _predict_frame(df: pd.DataFrame) -> List[Prediction]:
    try:
        proba = model.predict_proba(df)[:, 1]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f'Не удалось обработать визит: {exc}')

    session_ids = df.get('session_id', pd.Series([None] * len(df)))
    return [
        Prediction(
            session_id=None if pd.isna(sid) else str(sid),
            prediction=int(p >= THRESHOLD),
            probability=round(float(p), 6),
            threshold=THRESHOLD,
        )
        for sid, p in zip(session_ids, proba)
    ]


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
