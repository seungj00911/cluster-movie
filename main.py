
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.caption("영화의 흥행 규모와 롱런 특성으로 비슷한 영화끼리 묶습니다.")

FEATURES = {
    "스크린 수(로그)": "log_first_scrn",
    "누적 관객(로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}

try:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

raw_count = len(df)

for col in ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["log_first_scrn"] = np.where(
    df["first_scrn"] > 0,
    np.log10(df["first_scrn"]),
    np.nan
)
df["log_total_audi"] = np.where(
    df["total_audi"] > 0,
    np.log10(df["total_audi"]),
    np.nan
)
df["long_run_index"] = np.where(
    df["first_week_audi"] > 0,
    np.minimum(df["total_audi"] / df["first_week_audi"], 20),
    np.nan
)

feature_cols = list(FEATURES.values())

valid = df.dropna(
    subset=feature_cols + ["first_week_audi"]
).copy()

valid = valid[
    (valid["first_week_audi"] > 0)
    & (valid["first_scrn"] > 0)
    & (valid["total_audi"] > 0)
]

st.write(
    f"전체 영화 **{raw_count:,}편** · "
    f"군집화 대상 영화 **{len(valid):,}편**"
)

if len(valid) < 3:
    st.error("군집화를 하려면 분석 가능한 영화가 최소 3편 필요합니다.")
    st.stop()

st.subheader("1. 군집화 속성 선택")

selected_names = st.multiselect(
    "두 개 이상 선택하세요.",
    list(FEATURES.keys()),
    default=list(FEATURES.keys())
)

selected_features = [FEATURES[name] for name in selected_names]

if len(selected_features) < 2:
    st.warning("군집화 속성을 두 개 이상 선택하세요.")
    st.stop()

scaler = StandardScaler()
X = scaler.fit_transform(valid[selected_features])

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

valid["cluster_raw"] = model.fit_predict(X)

cluster_order = (
    valid.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

labels = ["㉮", "㉯", "㉰"]
label_map = {
    cluster_id: labels[i]
    for i, cluster_id in enumerate(cluster_order)
}

valid["유형"] = valid["cluster_raw"].map(label_map)

st.subheader("2. 2차원 산점도")

col1, col2 = st.columns(2)

with col1:
    x_name = st.selectbox(
        "가로축",
        selected_names,
        index=0
    )

with col2:
    y_name = st.selectbox(
        "세로축",
        selected_names,
        index=1
    )

fig2d = px.scatter(
    valid,
    x=FEATURES[x_name],
    y=FEATURES[y_name],
    color="유형",
    hover_name="movieNm",
    hover_data=["total_audi"],
    category_orders={"유형": ["㉮", "㉯", "㉰"]},
    labels={
        FEATURES[x_name]: x_name,
        FEATURES[y_name]: y_name,
        "유형": "영화 유형",
        "total_audi": "누적 관객"
    }
)

fig2d.update_traces(marker={"size": 7})
st.plotly_chart(fig2d, use_container_width=True)

st.subheader("3. 3차원 산점도")

if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 군집화 속성을 세 개 이상 선택하세요.")
else:
    c1, c2, c3 = st.columns(3)

    with c1:
        x3 = st.selectbox(
            "X축",
            selected_names,
            index=0,
            key="x3"
        )

    with c2:
        y3 = st.selectbox(
            "Y축",
            selected_names,
            index=1,
            key="y3"
        )

    with c3:
        z3 = st.selectbox(
            "Z축",
            selected_names,
            index=2,
            key="z3"
        )

    fig3d = px.scatter_3d(
        valid,
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

    fig3d.update_traces(marker={"size": 3})
    st.plotly_chart(fig3d, use_container_width=True)

st.subheader("4. 유형별 평균")

summary = valid.groupby("유형").agg(
    편수=("movieNm", "size"),
    스크린수평균=("first_scrn", "mean"),
    누적관객평균=("total_audi", "mean"),
    10위권일수평균=("days_in_top10", "mean"),
    롱런지수평균=("long_run_index", "mean")
).reindex(["㉮", "㉯", "㉰"]).reset_index()

summary = summary.rename(columns={
    "유형": "영화 유형",
    "편수": "영화 편수",
    "스크린수평균": "스크린 수 평균",
    "누적관객평균": "누적 관객 평균",
    "10위권일수평균": "10위권 일수 평균",
    "롱런지수평균": "롱런 지수 평균"
})

st.dataframe(
    summary.style.format({
        "스크린 수 평균": "{:,.2f}",
        "누적 관객 평균": "{:,.2f}",
        "10위권 일수 평균": "{:,.2f}",
        "롱런 지수 평균": "{:.2f}"
    }),
    hide_index=True,
    use_container_width=True
)

st.subheader("5. 유형별 누적 관객 상위 영화 5편")

for label in ["㉮", "㉯", "㉰"]:
    group = (
        valid[valid["유형"] == label]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    st.markdown(f"**{label} 유형**")

    if group.empty:
        st.write("해당 유형에 영화가 없습니다.")
    else:
        top5 = group[["movieNm", "total_audi"]].rename(
            columns={
                "movieNm": "영화 제목",
                "total_audi": "누적 관객"
            }
        ).copy()

        top5["누적 관객"] = top5["누적 관객"].map(
            lambda x: f"{x:,.0f}"
        )

        st.dataframe(
            top5,
            hide_index=True,
            use_container_width=True
        )
