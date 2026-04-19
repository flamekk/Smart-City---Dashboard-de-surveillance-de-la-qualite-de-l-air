from __future__ import annotations

"""Streamlit dashboard for monitoring the simulated IoT pipeline in real time."""

from pathlib import Path
import sys
from typing import Any, Dict, List

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Ensure imports like `from src...` work when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_nested, load_settings, resolve_path
from src.utils.helpers import read_jsonl_tail


TECH_STATUS_COLORS = {
    "normal": "#2E8B57",
    "warning": "#E6A700",
    "critical": "#CC2A2A",
    "unknown": "#5E5E5E",
}

SERIES_META: Dict[str, Dict[str, str]] = {
    "measurements_pm25": {"label": "PM2.5", "color": "#D62728"},
    "measurements_pm10": {"label": "PM10", "color": "#FF7F0E"},
    "measurements_co2_equivalent": {"label": "CO2 equivalent", "color": "#1F77B4"},
    "risk_score": {"label": "Risk score", "color": "#8A3FFC"},
    "measurements_temperature": {"label": "Temperature", "color": "#2CA02C"},
    "measurements_humidity": {"label": "Humidity", "color": "#17BECF"},
    "aqi_estimated": {"label": "AQI", "color": "#7F1D1D"},
    "ml_prediction_predicted_pm25": {"label": "Predicted PM2.5", "color": "#C2185B"},
    "ml_prediction_predicted_aqi": {"label": "Predicted AQI", "color": "#283593"},
}

AQI_META: Dict[str, Dict[str, str]] = {
    "good": {
        "label": "Good",
        "color": "#2E8B57",
        "icon": "&#9989;",
        "message": "Qualite de l'air satisfaisante avec un risque sanitaire faible.",
    },
    "moderate": {
        "label": "Moderate",
        "color": "#E6A700",
        "icon": "&#9888;",
        "message": "Qualite acceptable, avec un risque possible pour les groupes sensibles.",
    },
    "unhealthy_sensitive": {
        "label": "Unhealthy for Sensitive Groups",
        "color": "#FF8C00",
        "icon": "&#9888;",
        "message": "Qualite degradee: risque eleve pour les populations sensibles.",
    },
    "unhealthy": {
        "label": "Unhealthy",
        "color": "#CC2A2A",
        "icon": "&#9940;",
        "message": "Qualite mauvaise: effets sanitaires possibles pour la population generale.",
    },
    "very_unhealthy": {
        "label": "Very Unhealthy",
        "color": "#7B1FA2",
        "icon": "&#9940;",
        "message": "Episode de pollution tres eleve: risque sanitaire fort pour tous.",
    },
    "hazardous": {
        "label": "Hazardous",
        "color": "#5C1A1A",
        "icon": "&#9760;",
        "message": "Niveau dangereux: eviter l'exposition et declencher des actions de reponse.",
    },
    "unknown": {
        "label": "Unknown",
        "color": "#5E5E5E",
        "icon": "&#63;",
        "message": "Categorie AQI non disponible pour le moment.",
    },
}

ALERT_META: Dict[str, Dict[str, str]] = {
    "normal": {"label": "Normal", "color": "#2E8B57", "bg": "#EAF7EF", "icon": "&#10003;"},
    "warning": {"label": "Warning", "color": "#B36B00", "bg": "#FFF5E8", "icon": "&#9888;"},
    "critical": {"label": "Critical", "color": "#991B1B", "bg": "#FDECEC", "icon": "&#9940;"},
}


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .banner-sim {
            border: 1px solid #1f77b4;
            background: #eef6ff;
            border-radius: 10px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }
        .kpi-card {
            border: 1px solid #e5e7eb;
            border-left-width: 7px;
            border-radius: 10px;
            padding: 0.65rem 0.8rem;
            background: #ffffff;
            min-height: 88px;
        }
        .kpi-title { font-size: 0.85rem; color: #374151; margin-bottom: 0.2rem; }
        .kpi-value { font-size: 1.35rem; font-weight: 700; color: #111827; line-height: 1.2; }
        .kpi-sub { font-size: 0.75rem; color: #4b5563; margin-top: 0.2rem; }
        .status-card {
            border-radius: 10px;
            color: #ffffff;
            padding: 0.9rem 1rem;
            margin: 0.3rem 0 1rem 0;
        }
        .aqi-badge {
            color: #ffffff;
            font-weight: 700;
            font-size: 0.78rem;
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            margin-left: 0.4rem;
        }
        .alert-card {
            border-radius: 10px;
            border-left: 7px solid #999999;
            padding: 0.65rem 0.85rem;
            margin-bottom: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _normalize_category(raw_category: Any) -> str:
    if raw_category is None:
        return "unknown"
    normalized = str(raw_category).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "unhealthy_for_sensitive_groups": "unhealthy_sensitive",
        "sensitive": "unhealthy_sensitive",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in AQI_META else "unknown"


def _get_float(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(data.get(key, default))
    except (TypeError, ValueError):
        return default
    return value


def _load_alerts(alert_path: Path, limit: int = 60) -> pd.DataFrame:
    if not alert_path.exists():
        return pd.DataFrame(columns=["timestamp", "metric", "value", "threshold", "level", "message"])
    alerts_df = pd.read_csv(alert_path)
    if alerts_df.empty:
        return alerts_df
    if "timestamp" in alerts_df.columns:
        alerts_df["timestamp"] = pd.to_datetime(alerts_df["timestamp"], errors="coerce")
    return alerts_df.tail(limit).reset_index(drop=True)


def _extract_records(raw_payloads: List[Dict[str, Any]]) -> pd.DataFrame:
    if not raw_payloads:
        return pd.DataFrame()
    flattened = pd.json_normalize(raw_payloads, sep="_")
    if "timestamp" in flattened.columns:
        flattened["timestamp"] = pd.to_datetime(flattened["timestamp"], errors="coerce")
    return flattened


def _has_ml_predictions(df: pd.DataFrame) -> bool:
    required = {"ml_prediction_predicted_pm25", "ml_prediction_predicted_aqi", "ml_prediction_predicted_category"}
    return required.issubset(set(df.columns))


def _render_simulation_banner() -> None:
    st.markdown(
        """
        <div class="banner-sim">
            <div style="font-weight:700; color:#11467a;">Mode simulation actif</div>
            <div style="margin-top:0.2rem; color:#1f2937;">
                Les mesures sont diffusees depuis un dataset pour reproduire le comportement
                d'une chaine IoT en temps reel.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_global_status(latest: pd.Series) -> str:
    category_key = _normalize_category(latest.get("aqi_category", "unknown"))
    category_meta = AQI_META[category_key]
    technical_status = str(latest.get("status", "unknown")).lower()
    technical_color = TECH_STATUS_COLORS.get(technical_status, TECH_STATUS_COLORS["unknown"])

    st.markdown(
        (
            f"<div class='status-card' style='background:{category_meta['color']};'>"
            f"<div style='font-size:1.05rem; font-weight:700;'>{category_meta['icon']} "
            f"Qualite de l'air globale: {category_meta['label']}</div>"
            f"<div style='margin-top:0.25rem; font-size:0.9rem;'>{category_meta['message']}</div>"
            f"<div style='margin-top:0.45rem; font-size:0.78rem;'>"
            f"Etat systeme (alertes): <span style='background:{technical_color}; padding:0.15rem 0.45rem; "
            f"border-radius:999px; font-weight:700;'>{technical_status.upper()}</span>"
            f"</div></div>"
        ),
        unsafe_allow_html=True,
    )
    return category_key


def _kpi_card(title: str, value: str, color: str, subtitle: str = "") -> None:
    subtitle_html = f"<div class='kpi-sub'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        (
            f"<div class='kpi-card' style='border-left-color:{color};'>"
            f"<div class='kpi-title'>{title}</div>"
            f"<div class='kpi-value'>{value}</div>"
            f"{subtitle_html}"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_kpis(latest: pd.Series, category_key: str) -> None:
    latest_data = latest.to_dict()
    aqi_meta = AQI_META[category_key]
    aqi_value = int(_get_float(latest_data, "aqi_estimated", 0))
    aqi_badge = f"<span class='aqi-badge' style='background:{aqi_meta['color']};'>{aqi_meta['label']}</span>"

    row1 = st.columns(4)
    with row1[0]:
        _kpi_card("PM2.5 (ug/m3)", f"{_get_float(latest_data, 'measurements_pm25', 0.0):.2f}", SERIES_META["measurements_pm25"]["color"])
    with row1[1]:
        _kpi_card("PM10 (ug/m3)", f"{_get_float(latest_data, 'measurements_pm10', 0.0):.2f}", SERIES_META["measurements_pm10"]["color"])
    with row1[2]:
        _kpi_card(
            "AQI estime",
            f"{aqi_value} {aqi_badge}",
            SERIES_META["aqi_estimated"]["color"],
            "Categorie AQI affichee via un badge contextuel.",
        )
    with row1[3]:
        _kpi_card("Risk score (%)", f"{_get_float(latest_data, 'risk_score', 0.0):.1f}", SERIES_META["risk_score"]["color"])

    row2 = st.columns(3)
    with row2[0]:
        _kpi_card(
            "Temperature (C)",
            f"{_get_float(latest_data, 'measurements_temperature', 0.0):.2f}",
            SERIES_META["measurements_temperature"]["color"],
        )
    with row2[1]:
        _kpi_card(
            "Humidity (%)",
            f"{_get_float(latest_data, 'measurements_humidity', 0.0):.2f}",
            SERIES_META["measurements_humidity"]["color"],
        )
    with row2[2]:
        _kpi_card(
            "CO2 equivalent (ppm)",
            f"{_get_float(latest_data, 'measurements_co2_equivalent', 0.0):.1f}",
            SERIES_META["measurements_co2_equivalent"]["color"],
        )

    # Show optional ML inference KPIs only when prediction fields are present.
    has_ml = all(
        key in latest_data
        for key in ["ml_prediction_predicted_pm25", "ml_prediction_predicted_aqi", "ml_prediction_predicted_category"]
    )
    if not has_ml:
        return

    predicted_pm25 = _get_float(latest_data, "ml_prediction_predicted_pm25", 0.0)
    predicted_aqi = int(_get_float(latest_data, "ml_prediction_predicted_aqi", 0))
    predicted_category_key = _normalize_category(latest_data.get("ml_prediction_predicted_category"))
    predicted_category_meta = AQI_META[predicted_category_key]
    predicted_badge = (
        f"<span class='aqi-badge' style='background:{predicted_category_meta['color']};'>"
        f"{predicted_category_meta['label']}</span>"
    )

    measured_pm25 = _get_float(latest_data, "measurements_pm25", 0.0)
    pm25_gap = predicted_pm25 - measured_pm25
    gap_prefix = "+" if pm25_gap >= 0 else ""

    row3 = st.columns(3)
    with row3[0]:
        _kpi_card(
            "PM2.5 predit (ML)",
            f"{predicted_pm25:.2f}",
            SERIES_META["ml_prediction_predicted_pm25"]["color"],
            "Prediction du modele en ligne.",
        )
    with row3[1]:
        _kpi_card(
            "AQI predit (ML)",
            f"{predicted_aqi} {predicted_badge}",
            SERIES_META["ml_prediction_predicted_aqi"]["color"],
            "Categorie predite par le modele.",
        )
    with row3[2]:
        _kpi_card(
            "Ecart PM2.5 (pred - mesure)",
            f"{gap_prefix}{pm25_gap:.2f}",
            "#455A64",
            "Difference instantanee prediction vs mesure.",
        )


def _render_timeseries_chart(chart_df: pd.DataFrame, columns: List[str], title: str, y_title: str) -> None:
    available = [col for col in columns if col in chart_df.columns]
    if not available:
        st.info("Pas encore de donnees disponibles pour ce graphique.")
        return

    label_map = {col: SERIES_META.get(col, {}).get("label", col) for col in available}
    color_map = {label_map[col]: SERIES_META.get(col, {}).get("color", "#4B5563") for col in available}
    chart_source = chart_df[["timestamp"] + available].copy()
    melted = chart_source.melt(id_vars="timestamp", value_vars=available, var_name="series", value_name="value")
    melted["series"] = melted["series"].map(label_map)
    melted = melted.dropna(subset=["timestamp", "value"])

    if melted.empty:
        st.info("Pas encore de points numeriques exploitables pour ce graphique.")
        return

    chart = (
        alt.Chart(melted)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                title="Variable",
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Horodatage"),
                alt.Tooltip("series:N", title="Variable"),
                alt.Tooltip("value:Q", title="Valeur", format=".2f"),
            ],
        )
        .properties(height=300, title=title)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def _build_history_table(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    history_cols = [
        "timestamp",
        "status",
        "aqi_estimated",
        "aqi_category",
        "measurements_pm25",
        "measurements_pm10",
        "measurements_co2_equivalent",
        "ml_prediction_predicted_pm25",
        "ml_prediction_predicted_aqi",
        "ml_prediction_predicted_category",
        "measurements_temperature",
        "measurements_humidity",
        "risk_score",
    ]
    available = [col for col in history_cols if col in df.columns]
    if not available:
        return pd.DataFrame()

    view = df[available].tail(rows).iloc[::-1].copy()
    if "timestamp" in view.columns:
        view["timestamp"] = pd.to_datetime(view["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    if "status" in view.columns:
        view["status"] = view["status"].astype(str).str.upper()
    if "aqi_category" in view.columns:
        view["aqi_category"] = view["aqi_category"].map(lambda x: AQI_META[_normalize_category(x)]["label"])
    if "ml_prediction_predicted_category" in view.columns:
        view["ml_prediction_predicted_category"] = view["ml_prediction_predicted_category"].map(
            lambda x: AQI_META[_normalize_category(x)]["label"]
        )

    rename_map = {
        "timestamp": "Timestamp",
        "status": "System Status",
        "aqi_estimated": "AQI",
        "aqi_category": "AQI Category",
        "measurements_pm25": "PM2.5 (ug/m3)",
        "measurements_pm10": "PM10 (ug/m3)",
        "measurements_co2_equivalent": "CO2 equivalent (ppm)",
        "ml_prediction_predicted_pm25": "Pred PM2.5 (ML)",
        "ml_prediction_predicted_aqi": "Pred AQI (ML)",
        "ml_prediction_predicted_category": "Pred AQI Category (ML)",
        "measurements_temperature": "Temperature (C)",
        "measurements_humidity": "Humidity (%)",
        "risk_score": "Risk score (%)",
    }
    return view.rename(columns=rename_map)


def _render_alerts(alerts_df: pd.DataFrame, max_alerts: int) -> None:
    if alerts_df.empty:
        st.info("Aucune alerte recente.")
        return

    recent = alerts_df.tail(max_alerts).iloc[::-1].copy()
    recent["level"] = recent["level"].astype(str).str.lower()
    warning_count = int((recent["level"] == "warning").sum())
    critical_count = int((recent["level"] == "critical").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Warnings recentes", warning_count)
    c2.metric("Critical recentes", critical_count)
    c3.metric("Total alertes recentes", int(len(recent)))

    st.markdown("#### Evenements d'alerte")
    for _, row in recent.iterrows():
        level = row.get("level", "normal")
        meta = ALERT_META.get(level, ALERT_META["normal"])
        timestamp = row.get("timestamp", "")
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        metric = row.get("metric", "")
        value = row.get("value", "")
        threshold = row.get("threshold", "")
        message = row.get("message", "")
        st.markdown(
            (
                f"<div class='alert-card' style='background:{meta['bg']}; border-left-color:{meta['color']};'>"
                f"<div style='font-weight:700; color:{meta['color']};'>{meta['icon']} {meta['label']}</div>"
                f"<div style='font-size:0.82rem; color:#374151; margin-top:0.1rem;'>{timestamp}</div>"
                f"<div style='font-size:0.9rem; margin-top:0.28rem;'><b>Metric:</b> {metric} | "
                f"<b>Value:</b> {value} | <b>Threshold:</b> {threshold}</div>"
                f"<div style='font-size:0.9rem; margin-top:0.2rem;'>{message}</div>"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="Smart City Air Quality Dashboard", page_icon="AQ", layout="wide")
    _inject_css()
    settings = load_settings()

    refresh_seconds = int(get_nested(settings, "dashboard", "refresh_seconds", default=2))
    history_window = int(get_nested(settings, "dashboard", "history_window", default=150))
    data_log = resolve_path(get_nested(settings, "dashboard", "data_log_path", default="logs/stream_data.jsonl"))
    alert_log = resolve_path(get_nested(settings, "dashboard", "alert_log_path", default="logs/alerts_history.csv"))

    max_points = max(30, history_window)
    with st.sidebar:
        st.header("Controles d'affichage")
        pause_refresh = st.toggle("Pause rafraichissement auto", value=False)
        points_to_show = st.slider("Points recents (graphes)", min_value=30, max_value=max_points, value=min(120, max_points), step=10)
        history_rows = st.slider("Lignes historique", min_value=10, max_value=100, value=25, step=5)
        alerts_to_show = st.slider("Alertes recentes a afficher", min_value=5, max_value=30, value=12, step=1)
        show_ml_predictions = st.toggle("Afficher les predictions ML", value=True)
        selected_pollutants = st.multiselect(
            "Variables polluants/environnement",
            options=[
                "measurements_pm25",
                "measurements_pm10",
                "measurements_co2_equivalent",
                "measurements_temperature",
                "measurements_humidity",
            ],
            default=["measurements_pm25", "measurements_pm10", "measurements_co2_equivalent"],
            format_func=lambda x: SERIES_META[x]["label"],
        )

    if not pause_refresh:
        refresh_tick = st_autorefresh(interval=refresh_seconds * 1000, key="dashboard-refresh")
        st.caption(f"Rafraichissement auto actif toutes les {refresh_seconds}s | tick: {refresh_tick}")
    else:
        st.warning("Rafraichissement auto en pause. Desactive la pause pour reprendre le temps reel.")

    st.title("Smart City - Dashboard de surveillance de la qualite de l'air")
    _render_simulation_banner()

    records = read_jsonl_tail(data_log, limit=max(history_window, points_to_show))
    df = _extract_records(records)
    alerts_df = _load_alerts(alert_log)

    if df.empty:
        st.warning("Aucune donnee live. Lance le pipeline avec `python -m src.main`.")
        return

    ml_available = _has_ml_predictions(df)
    if show_ml_predictions and not ml_available:
        st.info("Predictions ML non detectees dans le flux. Lance le pipeline avec `python -m src.main --ml`.")
    if ml_available:
        st.caption("Inference ML active: le flux contient des champs de prediction (`ml_prediction_*`).")

    latest = df.iloc[-1]
    category_key = _render_global_status(latest)
    _render_kpis(latest, category_key)

    st.subheader("Tendances en temps reel")
    chart_data = df.tail(points_to_show).copy()
    pollutant_cols = [col for col in selected_pollutants if col in chart_data.columns]
    if not pollutant_cols:
        st.info("Aucune variable selectionnee n'est disponible dans le flux.")
    else:
        _render_timeseries_chart(
            chart_df=chart_data,
            columns=pollutant_cols,
            title="Polluants et indicateurs environnementaux",
            y_title="Concentration / valeur",
        )

    _render_timeseries_chart(
        chart_df=chart_data,
        columns=["risk_score", "aqi_estimated"],
        title="Evolution du score de risque et de l'AQI",
        y_title="Risque / AQI",
    )

    if show_ml_predictions and ml_available:
        st.subheader("Predictions ML")
        _render_timeseries_chart(
            chart_df=chart_data,
            columns=["measurements_pm25", "ml_prediction_predicted_pm25"],
            title="PM2.5 mesure vs PM2.5 predit (ML)",
            y_title="PM2.5 (ug/m3)",
        )
        _render_timeseries_chart(
            chart_df=chart_data,
            columns=["aqi_estimated", "ml_prediction_predicted_aqi"],
            title="AQI estime vs AQI predit (ML)",
            y_title="AQI",
        )

    st.subheader("Historique recent")
    history_table = _build_history_table(df, rows=history_rows)
    if history_table.empty:
        st.info("Aucune colonne d'historique disponible.")
    else:
        st.dataframe(history_table, use_container_width=True, hide_index=True)

    st.subheader("Alertes recentes")
    _render_alerts(alerts_df, max_alerts=alerts_to_show)


if __name__ == "__main__":
    main()
