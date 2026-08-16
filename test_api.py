"""
Проверка API без поднятия сервера: корректность ответов, устойчивость к пропускам
и неизвестным категориям, время ответа.

Запуск:
    python test_api.py
"""

import time

import pandas as pd
from fastapi.testclient import TestClient

import api

client = TestClient(api.app)
EXAMPLE = api.Visit.model_config['json_schema_extra']['example']


def check(name: str, condition: bool, extra: str = '') -> None:
    print(f'[{"ok " if condition else "FAIL"}] {name} {extra}')
    assert condition, name


# 1. сервис жив
r = client.get('/status')
check('GET /status', r.status_code == 200 and r.json()['status'] == 'ok')

# 2. метаданные модели
meta = client.get('/version').json()
check('GET /version отдаёт метрики', 'roc_auc' in meta,
      f"ROC-AUC={meta['roc_auc']:.4f}, порог={meta['threshold']:.4f}")

# 3. полный визит
r = client.post('/predict', json=EXAMPLE)
body = r.json()
check('POST /predict', r.status_code == 200 and body['prediction'] in (0, 1),
      f"prediction={body['prediction']}, probability={body['probability']:.4f}")

# 4. минимальный визит: только дата и время
r = client.post('/predict', json={'visit_date': '2021-07-01', 'visit_time': '12:00:00'})
check('минимальный запрос', r.status_code == 200, f"probability={r.json()['probability']:.4f}")

# 5. неизвестные категории не ломают модель
r = client.post('/predict', json={
    'visit_date': '2026-03-01', 'visit_time': '03:00:00',
    'utm_source': 'NEW_SOURCE_2026', 'utm_medium': 'tiktok',
    'geo_city': 'Atlantis', 'device_screen_resolution': 'битое значение',
    'device_category': 'mobile'})
check('неизвестные категории', r.status_code == 200, f"probability={r.json()['probability']:.4f}")

# 6. некорректная дата обрабатывается, а не роняет сервис
r = client.post('/predict', json={'visit_date': 'не дата', 'visit_time': 'не время'})
check('некорректная дата', r.status_code in (200, 422))

# 7. отсутствие обязательного поля
r = client.post('/predict', json={'visit_time': '12:00:00'})
check('валидация обязательных полей', r.status_code == 422)

# 8. батч
r = client.post('/predict_batch', json={'visits': [EXAMPLE] * 100})
check('POST /predict_batch', r.status_code == 200 and len(r.json()) == 100)

r = client.post('/predict_batch', json={'visits': []})
check('пустой батч отклоняется', r.status_code == 400)

# 9. время ответа
client.post('/predict', json=EXAMPLE)  # прогрев
t0 = time.time()
for _ in range(50):
    client.post('/predict', json=EXAMPLE)
single = (time.time() - t0) / 50

t0 = time.time()
client.post('/predict_batch', json={'visits': [EXAMPLE] * 1000})
batch = time.time() - t0

check('один визит быстрее 3 секунд', single < 3, f'{single * 1000:.0f} мс')
check('батч 1000 визитов быстрее 3 секунд', batch < 3, f'{batch * 1000:.0f} мс')

# 10. модель действительно различает визиты
probs = [client.post('/predict', json={**EXAMPLE, 'utm_medium': m}).json()['probability']
         for m in ['organic', 'stories', 'referral', 'cpc']]
check('прогноз зависит от канала', len(set(probs)) > 1,
      ', '.join(f'{m}={p:.4f}' for m, p in zip(['organic', 'stories', 'referral', 'cpc'], probs)))

print('\nВсе проверки пройдены.')
