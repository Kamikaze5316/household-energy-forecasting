from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Совместимость со старыми joblib/pickle-артефактами Scikit-learn.
# В некоторых версиях sklearn классы loss сериализовались под коротким
# именем модуля "_loss". В новых окружениях это имя может не регистрироваться
# автоматически, поэтому явно связываем его с текущим sklearn._loss._loss.
import sys
import sklearn
import sklearn.ensemble  # Инициализирует внутренние классы ансамблей.
try:
    from sklearn._loss import _loss as _sklearn_loss_module
    sys.modules.setdefault("_loss", _sklearn_loss_module)
except ImportError:
    # Для старых выпусков sklearn оставляем штатный механизм импорта.
    pass


# -----------------------------------------------------------------------------
# Конфигурация приложения и расположение артефактов
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Энергопотребление домохозяйства",
    page_icon="⚡",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
HISTORY_PATH = ARTIFACTS_DIR / "hourly_history.csv"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
TEST_METRICS_PATH = ARTIFACTS_DIR / "test_metrics.csv"
VALIDATION_METRICS_PATH = ARTIFACTS_DIR / "validation_metrics.csv"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"


# -----------------------------------------------------------------------------
# Загрузка сохранённых результатов
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str, modified_time: float):
    # Загружайте только артефакты, созданные вашим собственным ноутбуком.
    return joblib.load(model_path)


@st.cache_data
def load_metadata(metadata_path: str, modified_time: float) -> dict:
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_history(history_path: str, modified_time: float) -> pd.DataFrame:
    frame = pd.read_csv(history_path, parse_dates=["datetime"])
    frame = frame.set_index("datetime").sort_index()
    if "target_kw" not in frame.columns:
        raise ValueError("В hourly_history.csv отсутствует столбец target_kw.")
    frame.index.name = "datetime"
    return frame[["target_kw"]]


@st.cache_data
def load_metrics(metrics_path: str, modified_time: float) -> pd.DataFrame:
    return pd.read_csv(metrics_path, index_col=0)


@st.cache_data
def load_test_predictions(path: str, modified_time: float) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["datetime"])
    frame = frame.set_index("datetime").sort_index()
    required = {"actual_kw", "predicted_kw", "error_kw"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"В test_predictions.csv отсутствуют столбцы: {sorted(missing)}")
    return frame


def ensure_artifacts_exist() -> list[str]:
    required = [
        MODEL_PATH,
        HISTORY_PATH,
        METADATA_PATH,
        TEST_METRICS_PATH,
        VALIDATION_METRICS_PATH,
        TEST_PREDICTIONS_PATH,
    ]
    return [path.name for path in required if not path.exists()]


# -----------------------------------------------------------------------------
# Признаки и рекурсивное прогнозирование — те же определения, что в ноутбуке
# -----------------------------------------------------------------------------
def make_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    index = pd.DatetimeIndex(index)
    result = pd.DataFrame(index=index)

    hour = index.hour
    dayofweek = index.dayofweek
    month = index.month
    dayofyear = index.dayofyear

    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["weekday_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    result["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    result["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    result["dayofyear_sin"] = np.sin(2 * np.pi * (dayofyear - 1) / 365.25)
    result["dayofyear_cos"] = np.cos(2 * np.pi * (dayofyear - 1) / 365.25)
    result["is_weekend"] = (dayofweek >= 5).astype("int8")
    return result


def build_recursive_forecast(
    model,
    hourly_history: pd.DataFrame,
    metadata: dict,
    horizon: int,
) -> pd.Series:
    lags = [int(value) for value in metadata["lags"]]
    rolling_windows = [int(value) for value in metadata["rolling_windows"]]
    feature_columns = metadata["feature_columns"]
    required_history = max(max(lags), max(rolling_windows))

    history = hourly_history["target_kw"].asfreq("h").astype(float)
    # Соответствует обработке последних небольших разрывов в ноутбуке.
    history = history.interpolate(
        method="time",
        limit=3,
        limit_direction="both",
    )

    recent = history.tail(required_history)
    if len(recent) < required_history or recent.isna().any():
        raise ValueError(
            "Недостаточно непрерывной истории для прогноза. "
            f"Нужны последние {required_history} часа без пропусков. "
            "Проверьте последний участок исходного ряда."
        )

    recursive_history = history.copy()
    future_index = pd.date_range(
        start=recursive_history.index[-1] + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
    )
    predictions: list[float] = []

    for timestamp in future_index:
        row: dict[str, float] = {}

        for lag in lags:
            lag_timestamp = timestamp - pd.Timedelta(hours=lag)
            row[f"lag_{lag}"] = float(recursive_history.loc[lag_timestamp])

        for window in rolling_windows:
            previous_window = recursive_history.loc[
                timestamp - pd.Timedelta(hours=window):
                timestamp - pd.Timedelta(hours=1)
            ]
            if len(previous_window) != window or previous_window.isna().any():
                raise ValueError(
                    f"Недостаточно непрерывной истории для скользящего окна {window} часов."
                )
            row[f"rolling_mean_{window}"] = float(previous_window.mean())
            row[f"rolling_std_{window}"] = float(previous_window.std())

        row_frame = pd.DataFrame([row], index=pd.DatetimeIndex([timestamp]))
        row_frame = row_frame.join(make_calendar_features(row_frame.index))
        row_frame = row_frame[feature_columns]

        prediction = float(model.predict(row_frame)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        recursive_history.loc[timestamp] = prediction

    return pd.Series(predictions, index=future_index, name="forecast_kw")


def format_metric(value, suffix: str = "") -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.3f}{suffix}".replace(",", " ")


# -----------------------------------------------------------------------------
# Основной интерфейс
# -----------------------------------------------------------------------------
st.title("⚡ Прогнозирование энергопотребления домохозяйства")
st.caption(
    "Демонстрационное приложение на Streamlit · UCI Individual Household Electric Power Consumption"
)

missing_files = ensure_artifacts_exist()
if missing_files:
    st.error(
        "Не найдены файлы обученной модели и результатов. "
        "Сначала выполните обновлённый Google Colab Notebook, скачайте "
        "`streamlit_artifacts.zip`, распакуйте его в папку приложения и повторите запуск."
    )
    st.code("\n".join(f"artifacts/{name}" for name in missing_files))
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    metadata = load_metadata(str(METADATA_PATH), METADATA_PATH.stat().st_mtime)
    history = load_history(str(HISTORY_PATH), HISTORY_PATH.stat().st_mtime)
    test_metrics = load_metrics(str(TEST_METRICS_PATH), TEST_METRICS_PATH.stat().st_mtime)
    validation_metrics = load_metrics(
        str(VALIDATION_METRICS_PATH), VALIDATION_METRICS_PATH.stat().st_mtime
    )
    test_predictions = load_test_predictions(
        str(TEST_PREDICTIONS_PATH), TEST_PREDICTIONS_PATH.stat().st_mtime
    )
except ModuleNotFoundError as error:
    if error.name in {"_loss", "sklearn._loss", "numpy._core"}:
        st.error("Не удалось загрузить сериализованную модель из-за несовместимости окружений.")
        st.markdown(
            "Проверьте, что модель и приложение используют те же версии "
            "Scikit-learn, NumPy и XGBoost, что и при обучении в Google Colab. "
            "Обновлённый ноутбук сохраняет эти версии в `artifacts/metadata.json`, "
            "а `run_windows.bat` устанавливает их автоматически."
        )
        st.code(f"ModuleNotFoundError: {error.name}")
    else:
        st.exception(error)
    st.stop()
except Exception as error:
    st.exception(error)
    st.stop()

model_name = metadata.get("model_name", "Выбранная модель")

# Диагностика соответствия версий среды обучения и среды запуска.
training_environment = metadata.get("environment", {})
if training_environment:
    local_environment = {
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
    }
    mismatches = []
    for package_name, local_version in local_environment.items():
        trained_version = training_environment.get(package_name)
        if trained_version and trained_version != local_version:
            mismatches.append(f"{package_name}: обучение {trained_version}, приложение {local_version}")
    if mismatches:
        st.warning(
            "Версии библиотек отличаются от среды обучения. Для joblib-моделей "
            "Scikit-learn это может привести к ошибкам или изменению поведения. "
            "Перезапустите `run_windows.bat`, чтобы установить версии из metadata.json.\n\n"
            + "\n".join(f"- {item}" for item in mismatches)
        )

last_actual = history["target_kw"].dropna()
if last_actual.empty:
    st.error("В сохранённом историческом ряду нет корректных значений.")
    st.stop()

with st.sidebar:
    st.header("Параметры прогноза")
    horizon = st.slider("Горизонт прогноза, часов", min_value=1, max_value=72, value=24)
    history_days = st.selectbox("История на графике", options=[3, 7, 14, 30, 90], index=1)
    st.divider()
    st.caption("Модель обучена в Google Colab на подготовленном наборе UCI.")
    st.caption("Прогноз рекурсивный: предсказанные значения используются для следующих шагов.")

st.subheader("Сводка проекта")
metric_cols = st.columns(4)
metric_cols[0].metric("Выбранная модель", model_name)
metric_cols[1].metric("Последнее наблюдение", f"{last_actual.iloc[-1]:.3f} кВт")
metric_cols[2].metric("Последняя дата данных", last_actual.index[-1].strftime("%d.%m.%Y %H:%M"))
metric_cols[3].metric("Объём истории", f"{len(history):,} ч".replace(",", " "))

st.divider()
st.subheader(f"Прогноз на следующие {horizon} часов")

try:
    forecast = build_recursive_forecast(model, history, metadata, horizon)
except Exception as error:
    st.error(f"Не удалось построить прогноз: {error}")
    st.stop()

forecast_columns = st.columns(3)
forecast_columns[0].metric("Средняя прогнозная мощность", f"{forecast.mean():.3f} кВт")
forecast_columns[1].metric("Максимум за горизонт", f"{forecast.max():.3f} кВт")
forecast_columns[2].metric("Сумма прогнозных значений", f"{forecast.sum():.3f} кВт·ч")

plot_history = history["target_kw"].dropna().tail(history_days * 24)
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(plot_history.index, plot_history.values, label="Исторические значения", linewidth=1.2)
ax.plot(forecast.index, forecast.values, label="Прогноз", marker="o", markersize=3, linewidth=1.5)
ax.axvline(forecast.index[0], linestyle="--", alpha=0.7, label="Начало прогноза")
ax.set_title("Историческая динамика и прогноз энергопотребления")
ax.set_xlabel("Дата и время")
ax.set_ylabel("Средняя активная мощность, кВт")
ax.grid(True, alpha=0.25)
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

forecast_export = forecast.rename("forecast_kw").to_frame()
forecast_export.index.name = "datetime"
st.download_button(
    "Скачать прогноз (CSV)",
    data=forecast_export.to_csv(encoding="utf-8-sig").encode("utf-8-sig"),
    file_name=f"forecast_{horizon}_hours.csv",
    mime="text/csv",
)

with st.expander("Таблица прогноза"):
    forecast_table = forecast_export.copy()
    forecast_table.index = forecast_table.index.strftime("%d.%m.%Y %H:%M")
    forecast_table.index.name = "Дата и время"
    forecast_table = forecast_table.rename(columns={"forecast_kw": "Прогноз, кВт"})
    st.dataframe(forecast_table, use_container_width=True)

st.divider()
st.subheader("Качество моделей")
left, right = st.columns(2)
with left:
    st.markdown("**Валидационная выборка**")
    st.dataframe(validation_metrics.style.format(precision=3), use_container_width=True)
with right:
    st.markdown("**Тестовая выборка**")
    st.dataframe(test_metrics.style.format(precision=3), use_container_width=True)

if model_name in test_metrics.index:
    selected_metrics = test_metrics.loc[model_name]
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Test MAE", format_metric(selected_metrics.get("MAE, кВт"), " кВт"))
    kpi_cols[1].metric("Test RMSE", format_metric(selected_metrics.get("RMSE, кВт"), " кВт"))
    kpi_cols[2].metric("Test WAPE", format_metric(selected_metrics.get("WAPE, %"), "%"))
    kpi_cols[3].metric("Test R²", format_metric(selected_metrics.get("R²")))

st.subheader("Фактические и прогнозные значения на тестовом периоде")
preview_hours = min(history_days * 24, len(test_predictions))
preview = test_predictions.tail(preview_hours)
fig_test, ax_test = plt.subplots(figsize=(13, 4.5))
ax_test.plot(preview.index, preview["actual_kw"], label="Фактические значения", linewidth=1.1)
ax_test.plot(preview.index, preview["predicted_kw"], label="Прогноз модели", linewidth=1.1)
ax_test.set_title("Тестовый период: фактическая и прогнозная мощность")
ax_test.set_xlabel("Дата и время")
ax_test.set_ylabel("Мощность, кВт")
ax_test.grid(True, alpha=0.25)
ax_test.legend()
fig_test.autofmt_xdate()
fig_test.tight_layout()
st.pyplot(fig_test, use_container_width=True)
plt.close(fig_test)

st.download_button(
    "Скачать тестовые прогнозы (CSV)",
    data=test_predictions.to_csv(index_label="datetime", encoding="utf-8-sig").encode("utf-8-sig"),
    file_name="test_predictions.csv",
    mime="text/csv",
)

with st.expander("Методика и ограничения"):
    st.markdown(
        """
        **Постановка задачи.** Прогнозируется средняя активная мощность следующего часа на основе предыдущих измерений.

        **Признаки.** Используются лаги, скользящие средние и стандартные отклонения, а также календарные признаки.

        **Оценка качества.** Модели сравнивались на хронологически разделённых обучающей, валидационной и тестовой выборках. Тестовый период не участвовал в выборе модели.

        **Прогноз на несколько часов.** Применяется рекурсивная схема: каждый новый прогноз добавляется в историю и используется при вычислении признаков следующего часа. Поэтому ошибка может накапливаться с ростом горизонта.

        **Ограничения данных.** Набор описывает одно домохозяйство во Франции в 2006–2010 годах. Он не содержит информации о погоде, тарифах и составе жильцов, поэтому результаты нельзя напрямую переносить на другие домохозяйства.
        """
    )

st.caption("Источник данных: UCI Machine Learning Repository — Individual Household Electric Power Consumption (CC BY 4.0).")
