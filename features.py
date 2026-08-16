from __future__ import annotations

import numpy as np
import pandas as pd


# Справочники из брифа

# Целевые действия: любое из них в рамках визита означает конверсию
TARGET_ACTIONS = [
    'sub_car_claim_click',
    'sub_car_claim_submit_click',
    'sub_open_dialog_click',
    'sub_custom_question_submit_click',
    'sub_call_number_click',
    'sub_callback_submit_click',
    'sub_submit_success',
    'sub_car_request_submit_click',
]

# Органический трафик
ORGANIC_MEDIUMS = ['organic', 'referral', '(none)']

# Источники социальных сетей (зашифрованные ID из брифа)
SOCIAL_SOURCES = [
    'QxAxdyPLuQMEcrdZWdWb',
    'MvfHsxITijuriZxsqZqt',
    'ISrKoXQCxqqYvAZICvjs',
    'IZEXUFLARCUMynmHNBGo',
    'PlbkrSYoHuZBWfYjYnfw',
    'gVRrcxiDQubJiljoTbGm',
]

# Крупнейшие города, где сервис работает без доставки в регион
CAPITALS = ['Moscow', 'Saint Petersburg']

# Сырые поля визита, которые принимает модель на вход
RAW_COLUMNS = [
    'session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number',
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
    'device_category', 'device_os', 'device_brand', 'device_model',
    'device_screen_resolution', 'device_browser', 'geo_country', 'geo_city',
]

# Итоговые наборы признаков
CATEGORICAL_FEATURES = [
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword',
    'device_category', 'device_os', 'device_brand', 'device_browser',
    'geo_country', 'geo_city', 'traffic_type', 'day_part', 'os_family',
]

NUMERIC_FEATURES = [
    'visit_number', 'is_repeat_visit', 'hour', 'day_of_week', 'month',
    'day_of_month', 'week_of_year', 'is_weekend',
    'screen_width', 'screen_height', 'screen_area_log', 'screen_ratio',
    'is_organic', 'is_social', 'is_russia', 'is_capital',
    'is_mobile', 'has_utm_campaign', 'has_utm_keyword', 'is_paid_search',
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

MISSING = 'unknown'


# Очистка

def clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Базовая очистка таблицы визитов: пропуски, мусорные значения, дубликаты."""
    df = df.copy()

    # (not set), (none), (other) в GA означают отсутствие значения
    trash = {'(not set)', '(none)', '(other)', 'not set', 'nan', 'NaN', '', ' '}
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    for col in obj_cols:
        df[col] = df[col].replace(list(trash), np.nan)

    # utm_medium с (none) несёт смысл органики, возвращаем его обратно
    if 'utm_medium' in df.columns:
        df['utm_medium'] = df['utm_medium'].fillna('(none)')

    # device_model заполнен менее чем на 1 процент, признак бесполезен
    if 'device_model' in df.columns:
        df = df.drop(columns=['device_model'])

    if 'session_id' in df.columns:
        df = df.drop_duplicates(subset='session_id')

    return df


# Признаки

def _split_resolution(series: pd.Series) -> pd.DataFrame:
    res = series.fillna('0x0').astype(str).str.lower().str.replace(' ', '', regex=False)
    parts = res.str.split('x', n=1, expand=True)
    if parts.shape[1] == 1:
        parts[1] = '0'
    width = pd.to_numeric(parts[0], errors='coerce')
    height = pd.to_numeric(parts[1], errors='coerce')
    return pd.DataFrame({'screen_width': width, 'screen_height': height}, index=series.index)


def _day_part(hour: pd.Series) -> pd.Series:
    bins = [-1, 5, 11, 17, 23]
    labels = ['night', 'morning', 'day', 'evening']
    return pd.cut(hour.fillna(-1), bins=bins, labels=labels).astype(object).fillna(MISSING)


def _os_family(df: pd.DataFrame) -> pd.Series:
    os_col = df['device_os'] if 'device_os' in df.columns else pd.Series(np.nan, index=df.index)
    brand = df['device_brand'] if 'device_brand' in df.columns else pd.Series(np.nan, index=df.index)
    browser = df['device_browser'] if 'device_browser' in df.columns else pd.Series(np.nan, index=df.index)

    out = os_col.copy()
    # у 57 процентов визитов ОС не указана, восстанавливаем её по бренду и браузеру
    apple = brand.astype(str).str.lower().eq('apple') | browser.astype(str).str.startswith('Safari')
    out = out.mask(out.isna() & apple, 'iOS')
    android = brand.notna() & ~apple
    out = out.mask(out.isna() & android, 'Android')
    out = out.fillna(MISSING)

    mapping = {
        'iOS': 'Apple', 'Macintosh': 'Apple',
        'Android': 'Android', 'Windows': 'Windows',
        'Linux': 'Other', 'Chrome OS': 'Other',
    }
    return out.map(lambda v: mapping.get(v, 'Other' if v != MISSING else MISSING))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    "Превращает сырые поля визита в матрицу признаков модели."
    df = df.copy()

    for col in RAW_COLUMNS:
        if col not in df.columns and col != 'device_model':
            df[col] = np.nan

    # время визита
    date = pd.to_datetime(df['visit_date'], errors='coerce')
    time = pd.to_datetime(df['visit_time'], format='%H:%M:%S', errors='coerce')

    out = pd.DataFrame(index=df.index)
    out['hour'] = time.dt.hour.fillna(-1).astype('int16')
    out['day_of_week'] = date.dt.dayofweek.fillna(-1).astype('int16')
    out['month'] = date.dt.month.fillna(-1).astype('int16')
    out['day_of_month'] = date.dt.day.fillna(-1).astype('int16')
    out['week_of_year'] = date.dt.isocalendar().week.astype('float').fillna(-1).astype('int16')
    out['is_weekend'] = out['day_of_week'].isin([5, 6]).astype('int8')
    out['day_part'] = _day_part(out['hour'])

    # визит и посетитель
    visit_number = pd.to_numeric(df['visit_number'], errors='coerce').fillna(1)
    out['visit_number'] = visit_number.clip(upper=30).astype('int16')
    out['is_repeat_visit'] = (visit_number > 1).astype('int8')

    # трафик
    medium = df['utm_medium'].fillna('(none)').astype(str)
    source = df['utm_source'].fillna(MISSING).astype(str)
    out['is_organic'] = medium.isin(ORGANIC_MEDIUMS).astype('int8')
    out['is_social'] = source.isin(SOCIAL_SOURCES).astype('int8')
    out['is_paid_search'] = medium.isin(['cpc', 'cpm', 'cpa', 'cpv']).astype('int8')
    out['traffic_type'] = np.where(out['is_social'] == 1, 'social',
                          np.where(out['is_organic'] == 1, 'organic', 'paid'))
    out['has_utm_campaign'] = df['utm_campaign'].notna().astype('int8')
    out['has_utm_keyword'] = df['utm_keyword'].notna().astype('int8')

    # устройство
    res = _split_resolution(df['device_screen_resolution'])
    width = res['screen_width'].fillna(0).clip(0, 5000)
    height = res['screen_height'].fillna(0).clip(0, 5000)
    out['screen_width'] = width.astype('int16')
    out['screen_height'] = height.astype('int16')
    out['screen_area_log'] = np.log1p(width * height).astype('float32')
    out['screen_ratio'] = (height / width.replace(0, np.nan)).fillna(0).clip(0, 5).astype('float32')
    out['is_mobile'] = df['device_category'].fillna('').eq('mobile').astype('int8')
    out['os_family'] = _os_family(df)

    # гео
    country = df['geo_country'].fillna(MISSING).astype(str)
    city = df['geo_city'].fillna(MISSING).astype(str)
    out['is_russia'] = country.eq('Russia').astype('int8')
    out['is_capital'] = city.isin(CAPITALS).astype('int8')

    # категориальные как есть
    out['utm_source'] = source
    out['utm_medium'] = medium
    out['utm_campaign'] = df['utm_campaign'].fillna(MISSING).astype(str)
    out['utm_adcontent'] = df['utm_adcontent'].fillna(MISSING).astype(str)
    out['utm_keyword'] = df['utm_keyword'].fillna(MISSING).astype(str)
    out['device_category'] = df['device_category'].fillna(MISSING).astype(str)
    out['device_os'] = df['device_os'].fillna(MISSING).astype(str)
    out['device_brand'] = df['device_brand'].fillna(MISSING).astype(str)
    out['device_browser'] = df['device_browser'].fillna(MISSING).astype(str)
    out['geo_country'] = country
    out['geo_city'] = city

    return out[FEATURES]
