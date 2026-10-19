import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")

URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

FEATURES = {
    "스크린 수(로그)": "log_first_scrn",
    "누적 관객(로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index"
}

try:
    df = pd.read_csv(URL, encoding="utf-8")
except Exception as e:
    st.error(f"CSV 파일을 불러오지 못했습니다: {e}")
    st.stop()

required = [
    "movieNm", "first_scrn", "first_week_audi",
    "total_audi", "days_in_top10"
]

missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"데이터 열이 없습니다: {', '.join(missing)}")
    st.stop()

total_count = len(df)

for col in [
    "first_scrn", "first_week_audi",
    "total_audi", "days_in_top10"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["log_first_scrn"] = np.nan
df["log_total_audi"] = np.nan

mask_scrn = df["first_scrn"] > 0
mask_audi = df["total_audi"] > 0

df.loc[mask_scrn, "log_first_scrn"] = np.log10(
    df.loc[mask_scrn, "first_scrn"]
)
df.loc[mask_audi, "log_total_audi"] = np.log10(
    df.loc[mask_audi, "total_audi"]
)

df["long_run_index"] = np.nan
mask_week = df["first_week_audi"] > 0

df.loc[mask_week, "long_run_index"] = (
    df.loc[mask_week, "total_audi"]
    / df.loc[mask_week, "first_week_audi"]
).clip(upper=20)

feature_cols = list(FEATURES.values())

data = df.dropna(subset=feature_cols).copy()
data = data[
    (data["first_week_audi"] > 0)
    & (data["first_scrn"] > 0)
    & (data["total_audi"] > 0)
]

st.write(
    f"전체 영화: **{total_count:,}편** | "
    f"군집화 대상: **{len(data):,}편**"
)

if len(data) < 3:
    st.error("분석 가능한 영화가 3편 미만이라 군집화를 할 수 없습니다.")
    st.stop()

st.subheader("1. 군집화 속성")

selected_names = st.multiselect(
    "두 개 이상 선택하세요.",
    list(FEATURES.keys()),
    default=list(FEATURES.keys())
)

selected_cols = [FEATURES[x] for x in selected_names]

if len(selected_cols) < 2:
    st.warning("속성을 두 개 이상 선택해야 합니다.")
    st.stop()

scaler = StandardScaler()
X = scaler.fit_transform(data[selected_cols])

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
data["군집"] = model.fit_predict(X)

order = (
    data.groupby("군집")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

label_map = {
    cluster: label
    for cluster, label in zip(order, ["㉮", "㉯", "㉰"])
}
data["유형"] = data["군집"].map(label_map)

st.subheader("2. 2차원 산점도")

c1, c2 = st.columns(2)

with c1:
    x_name = st.selectbox("가로축", selected_names, index=0)

with c2:
    y_name = st.selectbox("세로축", selected_names, index=1)

fig2 = px.scatter(
    data,
    x=FEATURES[x_name],
    y=FEATURES[y_name],
    color="유형",
    hover_name="movieNm",
    category_orders={"유형": ["㉮", "㉯", "㉰"]},
    labels={
        FEATURES[x_name]: x_name,
        FEATURES[y_name]: y_name,
        "유형": "영화 유형"
    }
)
fig2.update_traces(marker={"size": 6})
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. 3차원 산점도")

if len(selected_names) < 3:
    st.info("3차원 산점도를 보려면 속성을 세 개 이상 선택하세요.")
else:
    a, b, c = st.columns(3)

    with a:
        x3 = st.selectbox("X축", selected_names, index=0, key="x3")
    with b:
        y3 = st.selectbox("Y축", selected_names, index=1, key="y3")
    with c:
        z3 = st.selectbox("Z축", selected_names, index=2, key="z3")

    fig3 = px.scatter_3d(
        data,
        x=FEATURES[x3],
        y=FEATURES[y3],
        z=FEATURES[z3],
        color="유형",
        hover_name="movieNm",
        category_orders={"유형": ["㉮", "㉯", "㉰"]},
        labels={
            FEATURES[x3]: x3,
            FEATURES[y3]: y3,
            FEATURES[z3]: z3,
            "유형": "영화 유형"
        }
    )
    fig3.update_traces(marker={"size": 3})
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("4. 유형별 평균")

summary = data.groupby("유형").agg(
    편수=("movieNm", "size"),
    스크린_수_평균=("first_scrn", "mean"),
    누적_관객_평균=("total_audi", "mean"),
    10위권_일수_평균=("days_in_top10", "mean"),
    롱런_지수_평균=("long_run_index", "mean")
).reindex(["㉮", "㉯", "㉰"]).reset_index()

summary.columns = [
    "유형", "편수", "스크린 수 평균",
    "누적 관객 평균", "10위권 일수 평균", "롱런 지수 평균"
]

st.dataframe(
    summary.style.format({
        "편수": "{:,.0f}",
        "스크린 수 평균": "{:,.2f}",
        "누적 관객 평균": "{:,.2f}",
        "10위권 일수 평균": "{:.2f}",
        "롱런 지수 평균": "{:.2f}"
    }),
    hide_index=True,
    use_container_width=True
)

st.subheader("5. 유형별 누적 관객 상위 영화")

for label in ["㉮", "㉯", "㉰"]:
    st.markdown(f"**{label} 유형**")
    top = (
        data[data["유형"] == label]
        .sort_values("total_audi", ascending=False)
        .head(5)[["movieNm", "total_audi"]]
        .rename(columns={
            "movieNm": "영화 제목",
            "total_audi": "누적 관객"
        })
        .copy()
    )

    if top.empty:
        st.write("영화가 없습니다.")
    else:
        top["누적 관객"] = top["누적 관객"].map(
            lambda x: f"{x:,.0f}"
        )
        st.dataframe(top, hide_index=True, use_container_width=True)
