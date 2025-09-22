# streamlit_app.py
# -*- coding: utf-8 -*-
"""
Streamlit + GitHub Codespaces 데이터 대시보드 (기후위기 정신건강 확장)

구성:
1) 공식 공개 데이터 대시보 (NASA POWER 일일 기온 API, 서울 좌표)
   - (기존과 동일)
2) 사용자 입력 대시보드 (프롬프트의 "폭염일수" 표 고정 내장)
   - (기존과 동일)
3) 기후위기 & 청소년 정신건강 (연구 참고) 탭 (★새로운 메뉴)
   - 기후위기/기온 상승과 우울증/불안 간의 상관관계 연구 결과 인용 및 요약 시각화
   - 한국 청소년 정신건강 현황(KYRBS 등 통계)을 간접 지표로 제시
   - 인과 관계 단정 불가 명시

폰트:
- /fonts/Pretendard-Bold.ttf 존재 시 Streamlit/Plotly에 적용 시도(없으면 자동 생략)

데이터 출처(코드 주석):
- NASA POWER API (일일 기상자료: 일 평균기온 T2M, 일 최고기온 T2M_MAX)
  https://power.larc.nasa.gov/docs/services/api/
- 참고 연구(청소년 자살충동 1°C당 1.3% 증가):
  PubMed: https://pubmed.ncbi.nlm.nih.gov/39441101/
- 추가 연구(기온/폭염 & 우울증/불안):
  Journal of Affective Disorders (중국 청소년): https://doi.org/10.1016/j.jad.2024.03.042
  PubMed (한국 성인): https://pubmed.ncbi.nlm.nih.gov/39242044/
"""

import io
import json
import math
import textwrap
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta
import plotly.express as px # Plotly를 전역으로 가져와서 사용

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="기온·폭염 & 청소년 정신건강(연구참고) 대시보드", layout="wide")

# Pretendard 적용 시도 (없으면 자동 생략)
def inject_font_css():
    font_path = Path("/fonts/Pretendard-Bold.ttf")
    if font_path.exists():
        st.markdown(
            f"""
            <style>
            @font-face {{
                font-family: 'Pretendard';
                src: url('file://{font_path.as_posix()}') format('truetype');
                font-weight: 700;
                font-style: normal;
            }}
            html, body, [class*="css"], .stMarkdown, .stButton, .stSelectbox, .stSlider, .stText, .stMetric, .stDataFrame {{
                font-family: 'Pretendard', 'Noto Sans KR', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

inject_font_css()

PLOTLY_FONT = "Pretendard, Noto Sans KR, Arial, sans-serif"

# 유틸
KST_TODAY = datetime.now()  # Codespaces는 UTC일 수 있으나, 미래 데이터 제거를 위해 절대 시점만 활용
TODAY_DATE = KST_TODAY.date()

def to_date(s):
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        try:
            return datetime.strptime(str(s), "%Y%m%d").date()
        except Exception:
            return pd.NaT

def clamp_to_today(df, date_col="date"):
    if df.empty:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    return df[df[date_col] <= TODAY_DATE]

def clean_standardize(df, date_col="date", value_col="value", group_col=None):
    df = df.copy()
    # 결측/중복 처리
    df = df.dropna(subset=[date_col])
    if group_col:
        df = df.drop_duplicates(subset=[date_col, group_col])
    else:
        df = df.drop_duplicates(subset=[date_col])
    # 타입 통일
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    # value를 숫자형으로
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    # 미래 데이터 제거
    df = clamp_to_today(df, date_col)
    return df

def download_button_for_df(df, filename, label="CSV 다운로드"):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# -----------------------------
# 1) 공개 데이터 대시보드
# (기존 코드 유지: fetch_nasa_power_daily, make_heatwave_flags, monthly_summary, plot_line, plot_bar, add_risk_annotation)
# -----------------------------
@st.cache_data(show_spinner=True, ttl=60 * 60)
def fetch_nasa_power_daily(lat=37.5665, lon=126.9780, start="2015-01-01", end=None):
    """ NASA POWER 일일 기온 데이터 가져오기 """
    if end is None:
        end = TODAY_DATE.strftime("%Y-%m-%d")

    start_str = pd.to_datetime(start).strftime("%Y%m%d")
    end_str = pd.to_datetime(end).strftime("%Y%m%d")

    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,T2M_MAX",
        "community": "RE",
        "latitude": lat,
        "longitude": lon,
        "start": start_str,
        "end": end_str,
        "format": "JSON",
    }
    try:
        r = requests.get(base_url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        t2m = js["properties"]["parameter"]["T2M"]
        t2m_max = js["properties"]["parameter"]["T2M_MAX"]
        records = []
        for k, v in t2m.items():
            d = to_date(k)
            if pd.isna(d):
                continue
            records.append({"date": d, "t2m": v, "t2m_max": t2m_max.get(k, np.nan)})
        df = pd.DataFrame(records)
        df = df.sort_values("date")
        # 표준화
        out = df.rename(columns={"t2m": "value"}).copy()
        out["group"] = "일 평균기온(℃)"
        out2 = df.rename(columns={"t2m_max": "value"}).copy()
        out2["group"] = "일 최고기온(℃)"
        all_df = pd.concat([out[["date", "value", "group"]], out2[["date", "value", "group"]]], ignore_index=True)
        all_df = clean_standardize(all_df, "date", "value", "group")
        all_df["fallback"] = False
        return all_df
    except Exception:
        # Fallback: 간단한 예시 데이터 생성 (최근 60일, 임의 패턴)
        dates = pd.date_range(end=TODAY_DATE, periods=60, freq="D")
        np.random.seed(42)
        base = 27 + np.sin(np.linspace(0, 3 * np.pi, len(dates))) * 5
        noise = np.random.normal(0, 1.2, len(dates))
        avg = base + noise
        tmax = avg + np.random.uniform(3, 8, len(dates))
        df = pd.DataFrame({"date": dates.date, "value": np.r_[avg, tmax], "group": ["일 평균기온(℃)"] * len(dates) + ["일 최고기온(℃)"] * len(dates)})
        df = clean_standardize(df, "date", "value", "group")
        df["fallback"] = True
        return df

def make_heatwave_flags(df, threshold_max=33.0):
    """ 폭염일 플래그 생성 """
    if df.empty:
        return df
    df = df.copy()
    w = df.pivot_table(index="date", columns="group", values="value")
    w["폭염일"] = (w.get("일 최고기온(℃)", pd.Series(index=w.index)) >= threshold_max).astype(int)
    out = (
        w.reset_index()[["date", "폭염일"]]
        .rename(columns={"폭염일": "value"})
        .assign(group=f"폭염일(최고기온≥{threshold_max}℃)")
    )
    return clean_standardize(out, "date", "value", "group")

def monthly_summary(df):
    """ 월별 합계/평균 요약 """
    if df.empty:
        return df
    x = df.copy()
    x["year"] = pd.to_datetime(x["date"]).dt.year
    x["month"] = pd.to_datetime(x["date"]).dt.month
    def agg_fn(g):
        if g.name[2].startswith("폭염일"):
            return pd.Series({"value": g["value"].sum()})
        else:
            return pd.Series({"value": g["value"].mean()})
    m = (
        x.groupby(["year", "month", "group"], as_index=False)
         .apply(agg_fn)
         .reset_index(drop=True)
    )
    m["date"] = pd.to_datetime(dict(year=m["year"], month=m["month"], day=1)).dt.date
    m = m[["date", "value", "group", "year", "month"]]
    return m

def plot_line(df, title, yaxis_title):
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    fig = px.line(
        df,
        x="date",
        y="value",
        color="group",
        markers=True,
        title=title,
    )
    fig.update_layout(
        xaxis_title="날짜",
        yaxis_title=yaxis_title,
        legend_title="지표",
        font=dict(family=PLOTLY_FONT),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_bar(df, title, yaxis_title, barmode="group"):
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    fig = px.bar(
        df,
        x="date",
        y="value",
        color="group",
        title=title,
        barmode=barmode,
    )
    fig.update_layout(
        xaxis_title="월",
        yaxis_title=yaxis_title,
        legend_title="지표",
        font=dict(family=PLOTLY_FONT),
        hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True)

def add_risk_annotation():
    st.markdown(
        """
        > 참고: **연구에 따르면, 하루 평균기온이 1°C 높아질 때마다 청소년(12~24세) 자살 충동/행동으로 인한 응급실 방문이 약 1.3% 증가**하는 경향이 관찰되었습니다.  
        > (호주 뉴사우스웨일스州, 2012–2019 시계열 분석. 인과 단정 불가, 참고 지표로만 활용)
        """
    )
    with st.expander("연구 출처(주석) 보기", expanded=False):
        st.code(
            textwrap.dedent(
                """
                PubMed (청소년 자살충동 1°C당 1.3% 증가):
                https://pubmed.ncbi.nlm.nih.gov/39441101/
                """
            ),
            language="text",
        )

# -----------------------------
# 2) 사용자 입력 대시보드 데이터
# (기존 코드 유지: load_user_table, plot_user_monthly, plot_user_rank)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_user_table():
    """ 프롬프트에 포함된 '폭염일수' 표를 내장 CSV로 구성. """
    raw = """연도,1월,2월,3월,4월,5월,6월,7월,8월,9월,10월,11월,12월,연합계,순위
2015,0,0,0,0,0,1,4,3,0,0,0,0,8,10
2016,0,0,0,0,0,0,4,20,0,0,0,0,24,4
2017,0,0,0,0,0,1,5,7,0,0,0,0,13,8
2018,0,0,0,0,0,0,16,19,0,0,0,0,35,1
2019,0,0,0,0,1,0,4,10,0,0,0,0,15,7
2020,0,0,0,0,0,2,0,2,0,0,0,0,4,11
2021,0,0,0,0,0,0,15,3,0,0,0,0,18,6
2022,0,0,0,0,0,0,10,0,0,0,0,0,10,9
2023,0,0,0,0,0,2,6,11,0,0,0,0,19,5
2024,0,0,0,0,0,4,2,21,6,0,0,0,33,2
2025,0,0,0,0,0,3,15,9,1,,,,28,3
평균,0.0,0.0,0.0,0.0,0.1,1.2,7.4,9.6,0.6,0.0,0.0,0.0,,  
"""
    df = pd.read_csv(io.StringIO(raw))
    # "평균" 행 제거
    df = df[df["연도"].apply(lambda x: str(x).isdigit())].copy()
    df["연도"] = df["연도"].astype(int)

    # melt 월별
    month_cols = ["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월","11월","12월"]
    keep_cols = ["연도","연합계","순위"]
    for c in month_cols:
        if c not in df.columns:
            df[c] = np.nan

    m = df.melt(id_vars=keep_cols + ["연도"], value_vars=month_cols, var_name="월", value_name="폭염일수")
    # 날짜 생성: 각 월의 1일
    m["월_int"] = m["월"].str.replace("월", "", regex=False).astype(int)
    m["date"] = pd.to_datetime(dict(year=m["연도"], month=m["월_int"], day=1)).dt.date
    m["value"] = pd.to_numeric(m["폭염일수"], errors="coerce")

    # 표준화 date, value, group(연도)
    out = m[["date", "value", "연도"]].rename(columns={"연도": "group"})
    out = clean_standardize(out, "date", "value", "group")
    # 미래 월 제거
    out = clamp_to_today(out, "date")

    # 연도별 연합계/순위 테이블도 보관
    yr = df[["연도", "연합계", "순위"]].rename(columns={"연도":"year","연합계":"total","순위":"rank"})
    yr["total"] = pd.to_numeric(yr["total"], errors="coerce")
    yr["rank"] = pd.to_numeric(yr["rank"], errors="coerce")
    return out, yr

def plot_user_monthly(df_long):
    if df_long.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    fig = px.line(
        df_long,
        x="date",
        y="value",
        color="group",
        markers=True,
        title="연도별 월간 폭염일수 추이",
    )
    fig.update_layout(
        xaxis_title="월",
        yaxis_title="폭염일수(일)",
        legend_title="연도",
        font=dict(family=PLOTLY_FONT),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_user_rank(yr):
    y2 = yr.dropna(subset=["year","total","rank"]).copy()
    if y2.empty:
        st.info("순위 데이터가 없습니다.")
        return
    y2["date"] = pd.to_datetime(dict(year=y2["year"], month=1, day=1)).dt.date
    # 순위는 낮을수록 상위이므로 y축 뒤집기
    fig = px.scatter(
        y2,
        x="year",
        y="rank",
        size="total",
        text="total",
        title="연도별 폭염일수 연합계 & 순위",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title="연도",
        yaxis_title="순위(낮을수록 상위)",
        yaxis=dict(autorange="reversed"),
        font=dict(family=PLOTLY_FONT),
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 3) 기후위기 & 청소년 정신건강 대시보드 (신규 추가)
# -----------------------------
@st.cache_data(show_spinner=False)
def get_mental_health_indicators():
    """
    청소년 정신건강 관련 주요 통계/연구 결과 요약 데이터를 반환
    실제 시계열 데이터 대신, 검색된 주요 연구 결과 및 한국 통계를 인용하여 구성
    """
    # 1. 기후위기 관련 연구 결과 (비교 지표)
    # 중국 청소년 폭염 vs 우울증/불안 (폭염 강도 1단위 증가당)
    # 한국 성인 (19-40세, 도시 거주) 기온 1°C 증가 vs 우울 증상 (1961-1990 대비)
    research_indicators = pd.DataFrame([
        {"지표": "폭염 vs 우울증 위험 증가", "단위": "%", "값": 13, "출처": "연구(중국 청소년)", "설명": "폭염 강도 1단위 증가당"},
        {"지표": "폭염 vs 불안 위험 증가", "단위": "%", "값": 12, "출처": "연구(중국 청소년)", "설명": "폭염 강도 1단위 증가당"},
        {"지표": "기온 1°C↑ vs 우울 증상 위험 증가", "단위": "%", "값": 14, "출처": "연구(한국 성인 19-40세)", "설명": "1961-1990 대비 연평균 기온 1°C 증가당"},
    ])

    # 2. 한국 청소년 정신건강 현황 (간접 지표, KYRBS 인용) - 예시 데이터로 구성
    # 출처: 청소년건강행태조사(KYRBS), 연도별 수치는 예시값
    kyrbs_data = pd.DataFrame({
        "연도": [2021, 2022, 2023, 2024, 2025], # 최근 5년으로 가정
        "우울감 경험률(%)": [25.0, 26.5, 27.2, 28.5, 29.1], # 예시값, 증가 추세 가정
        "자살 생각률(%)": [10.5, 11.0, 11.3, 11.5, 11.8], # 예시값, 증가 추세 가정
    })
    kyrbs_data["date"] = pd.to_datetime(dict(year=kyrbs_data["연도"], month=1, day=1)).dt.date
    kyrbs_data = clamp_to_today(kyrbs_data, "date")
    
    # 시각화를 위한 Melt
    melted_kyrbs = kyrbs_data.melt(
        id_vars=["연도", "date"], 
        value_vars=["우울감 경험률(%)", "자살 생각률(%)"],
        var_name="group", 
        value_name="value"
    ).rename(columns={"value": "value_perc"})

    return research_indicators, melted_kyrbs

def plot_kyrbs_trend(df):
    """ 청소년 정신건강 지표 추이 (KYRBS 기반 예시) """
    if df.empty:
        st.info("청소년 정신건강 현황 데이터가 없습니다.")
        return
    
    fig = px.line(
        df,
        x="연도",
        y="value_perc",
        color="group",
        markers=True,
        title="청소년 정신건강 주요 지표 추이 (가상 데이터, KYRBS 등 참고)",
    )
    fig.update_layout(
        xaxis_title="연도",
        yaxis_title="비율(%)",
        legend_title="지표",
        font=dict(family=PLOTLY_FONT),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# 사이드바
# -----------------------------
with st.sidebar:
    st.header("옵션")
    st.caption("※ 모든 라벨은 한국어, 오늘 이후 데이터는 자동 제거됩니다.")

# -----------------------------
# 탭 구성
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📡 공개 데이터 대시보드 (NASA POWER, 서울)", "📘 사용자 입력 대시보드 (폭염일수)", "🧠 기후위기 & 청소년 정신건강(연구참고)"])

with tab1:
    st.subheader("서울 일별 기온 & 폭염일 (NASA POWER)")
    st.caption("출처: NASA POWER API (T2M/T2M_MAX). API 실패 시 예시 데이터로 대체 표시됩니다.")

    colA, colB, colC = st.columns(3)
    with colA:
        start_date = st.date_input("조회 시작일", value=date(2015,1,1), min_value=date(1981,1,1), max_value=TODAY_DATE)
    with colB:
        end_date = st.date_input("조회 종료일", value=TODAY_DATE, min_value=start_date, max_value=TODAY_DATE)
    with colC:
        hw_threshold = st.number_input("폭염 기준(일최고기온, ℃)", min_value=30.0, max_value=40.0, value=33.0, step=0.5)

    data = fetch_nasa_power_daily(start=start_date.isoformat(), end=end_date.isoformat())
    if data["fallback"].any():
        st.warning("API 호출 실패로 예시 데이터가 표시됩니다. (네트워크/서비스 상태 확인 필요)")

    # 폭염일 플래그 시계열
    hw = make_heatwave_flags(data, threshold_max=hw_threshold)

    # 표준화 테이블 병합(기온 + 폭염일)
    std = pd.concat([data[["date","value","group"]], hw[["date","value","group"]]], ignore_index=True)
    std = clean_standardize(std, "date", "value", "group")

    # 기간 슬라이더(월 단위)
    if not std.empty:
        min_d = pd.to_datetime(std["date"]).min().date()
        max_d = pd.to_datetime(std["date"]).max().date()
        # 사이드바 옵션으로 이동
        with st.sidebar:
            st.markdown("#### 공개 데이터 기간 필터")
            rng = st.slider("표시 기간 선택", min_value=min_d, max_value=max_d, value=(min_d, max_d), key="tab1_rng")
            smooth_win = st.select_slider("이동평균 윈도우(일, 기온에만 적용)", options=[1,3,5,7,14], value=3, key="tab1_smooth")
        
        std = std[(std["date"] >= rng[0]) & (std["date"] <= rng[1])]

        # 스무딩(이동평균, 기온만)
        if smooth_win > 1 and not std.empty:
            gtemp = std["group"].isin(["일 평균기온(℃)","일 최고기온(℃)"])
            std.loc[gtemp, "value"] = (
                std[gtemp]
                .sort_values("date")
                .groupby("group")["value"]
                .transform(lambda s: s.rolling(smooth_win, min_periods=1).mean())
            )

    # 시각화
    plot_line(std[std["group"].isin(["일 평균기온(℃)", "일 최고기온(℃)"])], "일별 기온 추이", "기온(℃)")

    msum = monthly_summary(pd.concat([data[["date","value","group"]], hw], ignore_index=True))
    # 월별 폭염일수 & 월평균/월평균최고
    monthly_heat = msum[msum["group"].str.startswith("폭염일")]
    monthly_temp = msum[~msum["group"].str.startswith("폭염일")]

    plot_bar(monthly_heat, "월별 폭염일수(합계)", "폭염일수(일)")
    plot_line(monthly_temp, "월별 평균 기온/최고기온(평균)", "기온(℃)")

    # 참고 연구 안내
    add_risk_annotation()
    st.info(
        "※ 본 대시보드는 **기온·폭염과 정신건강 지표 간 상관성**에 대한 참고 탐색용입니다. "
        "인과관계를 단정하지 않으며, 지역·연령·제도 차이에 따라 결과 해석에 주의가 필요합니다."
    )

    # 다운로드(표준화 테이블)
    st.markdown("#### 전처리된 표 다운로드")
    download_button_for_df(std[["date","value","group"]].sort_values(["date","group"]), "nasa_power_standardized.csv", "CSV 다운로드 (공개 데이터)")

    # 주석으로 출처 URL 남김
    st.caption("주석: NASA POWER API 문서 URL은 코드 주석에 기재되어 있습니다. (앱 상단 주석 참조)")

with tab2:
    st.subheader("사용자 입력 데이터 대시보드 — 폭염일수(연도·월)")
    st.caption("프롬프트로 제공된 표만 사용합니다. 업로드나 추가 입력을 요구하지 않습니다.")

    user_long, user_year = load_user_table()

    # 사이드바/옵션
    if not user_long.empty:
        y_min = int(pd.to_datetime(user_long["date"]).dt.year.min())
        y_max = int(pd.to_datetime(user_long["date"]).dt.year.max())
        with st.sidebar:
            st.markdown("#### 사용자 데이터 기간/스무딩")
            y_start, y_end = st.slider("표시 연도 범위", min_value=y_min, max_value=y_max, value=(y_min, y_max), key="tab2_yr_rng")
            smooth_months = st.select_slider("월 이동평균(연도별 적용)", options=[1,3], value=1, key="tab2_smooth")

        view_df = user_long[(pd.to_datetime(user_long["date"]).dt.year >= y_start) & (pd.to_datetime(user_long["date"]).dt.year <= y_end)]
    else:
        view_df = user_long
        smooth_months = 1 # 옵션이 없으므로 기본값 설정

    # 스무딩(월 이동평균, 각 연도별)
    if smooth_months > 1 and not view_df.empty:
        view_df = view_df.sort_values(["group","date"]).copy()
        view_df["value"] = view_df.groupby("group")["value"].transform(lambda s: s.rolling(smooth_months, min_periods=1).mean())

    # 시각화
    plot_user_monthly(view_df)
    st.markdown("---")
    plot_user_rank(user_year)

    # 표준화 표 미리보기 & 다운로드
    st.markdown("#### 전처리된 표 (표준화: date, value, group)")
    st.dataframe(view_df.sort_values(["date","group"]), use_container_width=True)
    download_button_for_df(view_df.sort_values(["date","group"]), "user_heatdays_standardized.csv", "CSV 다운로드 (사용자 데이터)")

with tab3:
    st.subheader("기후위기 & 청소년 정신건강(연구 참고)")
    st.caption("기온 변화와 정신건강 지표의 상관관계 연구 결과 및 한국 청소년 현황 통계를 간접 지표로 활용합니다.")

    research_df, kyrbs_df = get_mental_health_indicators()

    # 1. 주요 연구 인용 요약
    st.markdown("#### 🌡️ 기온 변화와 정신건강 위험도 증가 (주요 연구 인용)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=f"{research_df.iloc[0]['지표']} ({research_df.iloc[0]['설명']})",
            value=f"+{research_df.iloc[0]['값']}{research_df.iloc[0]['단위']}",
            help=f"출처: {research_df.iloc[0]['출처']}"
        )
    with col2:
        st.metric(
            label=f"{research_df.iloc[1]['지표']} ({research_df.iloc[1]['설명']})",
            value=f"+{research_df.iloc[1]['값']}{research_df.iloc[1]['단위']}",
            help=f"출처: {research_df.iloc[1]['출처']}"
        )
    with col3:
        st.metric(
            label=f"{research_df.iloc[2]['지표']} ({research_df.iloc[2]['설명']})",
            value=f"+{research_df.iloc[2]['값']}{research_df.iloc[2]['단위']}",
            help=f"출처: {research_df.iloc[2]['출처']}"
        )

    st.markdown("---")
    
    # 2. 한국 청소년 정신건강 주요 지표 추이 (가상 데이터)
    st.markdown("#### 🇰🇷 한국 청소년 정신건강 현황 추이 (KYRBS 기반 예시)")
    plot_kyrbs_trend(kyrbs_df)

    # 3. 상세 연구 출처 및 유의 사항
    st.markdown("#### 💡 데이터 유의 사항 및 연구 출처")
    st.warning(
        "**주의:** 제시된 '한국 청소년 정신건강 현황 추이'는 실제 KYRBS 데이터의 대략적인 경향을 기반으로 **임의로 생성된 예시 데이터**입니다. "
        "기후변화와 정신건강 간의 **인과관계**가 아닌, **상관관계 및 잠재적 위험 증가**를 보여주는 참고 지표로만 활용해야 합니다."
    )
    with st.expander("참고 문헌 (주석)", expanded=False):
        st.markdown(
            """
            * **기존 연구 (청소년 자살충동 vs 기온):** PubMed: https://pubmed.ncbi.nlm.nih.gov/39441101/
            * **폭염 vs 우울증/불안 (중국 청소년):** Journal of Affective Disorders (2024). 폭염 강도 1단위 증가당 우울증 13%, 불안 12% 증가.
            * **기온 1°C↑ vs 우울 증상 (한국 성인 19-40세):** PubMed (2024). 연평균 기온 1°C 증가당 우울 증상 14% 증가 (도시 거주).
            * **한국 청소년 정신건강 현황:** 교육부/질병관리청 **청소년건강행태조사(KYRBS)**의 공표 통계(예: 우울감 경험률, 자살 생각률)를 참고하여 임의의 시계열 예시 데이터를 생성했습니다.
            """
        )

# 푸터
st.markdown("---")
st.caption("© Streamlit 대시보드 예시. 데이터는 공개 API/제공 표/연구 인용 기준으로 구성되며, 오늘(로컬 자정) 이후 데이터는 제거됩니다.")