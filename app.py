import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import re  # <-- dùng re.escape thay vì pd.regex

# ============================
# App Config & Header
# ============================
st.set_page_config(page_title="VN Jobs 2024 — Phân tích & Dự báo", layout="wide")
st.title("Phân tích, dự đoán và trực quan hóa dữ liệu tuyển dụng trực tuyến Việt Nam – 2024")
st.caption("Lương chuẩn hóa về **triệu VND/tháng** (USD → VND: giả định 1 USD ≈ 0.025 triệu VND).")

# ============================
# Utilities
# ============================
def strip_accents(s: str) -> str:
    """Remove Vietnamese accents and normalize spaces."""
    if s is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(s))
    s2 = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    s2 = " ".join(s2.split())
    return s2

def norm_text_series(sr: pd.Series) -> pd.Series:
    return sr.astype(str).map(strip_accents).str.lower().str.strip()

def norm_text_value(v) -> str:
    return strip_accents(str(v)).lower().strip()

# ============================
# Load Data  (NO CACHE → tránh MemoryError)
# ============================
def load_data():
    df = pd.read_csv("clean_jobs.csv")

    # Ensure columns
    needed_cols = ["salary_mid", "experience_years", "job_fields", "city", "position_level", "job_title"]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Numeric casts
    df["salary_mid"] = pd.to_numeric(df["salary_mid"], errors="coerce")
    df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce")

    # Original (for display)
    for c in ["job_fields", "city", "position_level", "job_title"]:
        df[c] = df[c].astype(str).str.strip()

    # Normalized (for filtering)
    df["city_norm"] = norm_text_series(df["city"])
    df["position_level_norm"] = norm_text_series(df["position_level"])
    df["job_fields_norm"] = norm_text_series(df["job_fields"])

    # Clean common noisy labels in level
    bad_levels = {"", "nan", "none", "error", "chua cap nhat", "chua cap nhat.", "chua cap nhat -", "chua cap nhat,"}
    df.loc[df["position_level_norm"].isin(bad_levels), "position_level_norm"] = ""

    return df

df = load_data()

# ============================
# Sidebar Filters
# ============================
st.sidebar.header("Bộ lọc dữ liệu tổng quan")

# Partial/exact toggles
copt1, copt2 = st.sidebar.columns(2)
with copt1:
    city_partial = st.toggle("City chứa", value=True, help="Bật: khớp 1 phần (contains). Tắt: khớp tuyệt đối.")
with copt2:
    level_partial = st.toggle("Cấp bậc chứa", value=True, help="Bật: khớp 1 phần (contains). Tắt: khớp tuyệt đối.")

# Suggestions
with st.sidebar.expander("Gợi ý giá trị (Top 30)"):
    if "city" in df.columns:
        st.write("**Thành phố**")
        st.dataframe(df["city"].str.lower().value_counts().head(30).to_frame("count"), width="stretch")
    if "position_level" in df.columns:
        st.write("**Cấp bậc**")
        st.dataframe(df["position_level"].str.lower().value_counts().head(30).to_frame("count"), width="stretch")

# Options from original text (user-friendly), filtering uses *_norm
city_options = sorted([x for x in df["city"].dropna().unique().tolist() if str(x).strip() and str(x).lower() != "nan"])
level_options = sorted([x for x in df["position_level"].dropna().unique().tolist() if str(x).strip() and str(x).lower() != "nan"])

cities = st.sidebar.multiselect("Thành phố", city_options, default=[])
levels = st.sidebar.multiselect("Cấp bậc", level_options, default=[])
fields_kw = st.sidebar.text_input("Lọc theo lĩnh vực (từ khóa, ví dụ: 'marketing, it')")

# Reset filters (an toàn kể cả khi không dùng cache)
if st.sidebar.button("🧹 Xoá lọc"):
    st.experimental_rerun()

def norm_set(values):
    return [norm_text_value(v) for v in values if str(v).strip()]

def apply_global_filters(dataframe: pd.DataFrame) -> pd.DataFrame:
    flt = dataframe

    # City
    if cities:
        want = norm_set(cities)
        if city_partial:
            patt = "|".join([re.escape(w) for w in want])  # <-- dùng re.escape
            flt = flt[flt["city_norm"].str.contains(patt, regex=True, na=False)]
        else:
            flt = flt[flt["city_norm"].isin(want)]

    # Level
    if levels:
        want = norm_set(levels)
        if level_partial:
            patt = "|".join([re.escape(w) for w in want])  # <-- dùng re.escape
            flt = flt[flt["position_level_norm"].str.contains(patt, regex=True, na=False)]
        else:
            flt = flt[flt["position_level_norm"].isin(want)]

    # Fields keywords (partial match on normalized)
    if fields_kw:
        kws = [norm_text_value(x) for x in fields_kw.split(",") if x.strip()]
        if kws:
            flt = flt[flt["job_fields_norm"].fillna("").apply(lambda s: any(kw in s for kw in kws))]

    return flt

filtered = apply_global_filters(df)

# ============================
# Tabs
# ============================
tab_viz, tab_hot, tab_prophet = st.tabs(["📊 Trực quan hoá", "🔮 Dự đoán ngành hot", "🧭 Dự báo (Prophet)"])

# ====== TAB 1: Visualization ======
with tab_viz:
    st.subheader("Tổng quan dữ liệu")
    st.markdown(f"Bản ghi **trước lọc**: `{len(df):,}` • **sau lọc**: `{len(filtered):,}`")

    st.markdown("### Top thành phố theo số tin đăng")
    if "city" in filtered.columns and not filtered.empty:
        st.bar_chart(filtered["city"].value_counts().head(15))
    else:
        st.info("Không có dữ liệu city sau khi lọc.")

    st.markdown("### Phân phối lương (triệu VND/tháng)")
    sal = pd.to_numeric(filtered.get("salary_mid", pd.Series(dtype=float)), errors="coerce").dropna()
    if not sal.empty:
        st.line_chart(sal)
    else:
        st.info("Không có dữ liệu lương hợp lệ sau khi lọc.")

# ====== TAB 2: Hot Fields Prediction ======
with tab_hot:
    st.subheader("Xếp hạng ngành *hot* theo lương & kinh nghiệm (bộ nhớ an toàn)")

    # Inputs
    sal_series = pd.to_numeric(df["salary_mid"], errors="coerce")
    sal_min = float(np.nanmin(sal_series)) if sal_series.notna().any() else 0.0
    sal_max = float(np.nanmax(sal_series)) if sal_series.notna().any() else 100.0
    default_sal = float(np.nanmedian(sal_series)) if sal_series.notna().any() else 10.0

    exp_series = pd.to_numeric(df.get("experience_years", pd.Series(dtype=float)), errors="coerce")
    exp_max = max(20.0, float(np.nanmax(exp_series)) if exp_series.notna().any() else 0.0)
    default_exp = float(np.nanmedian(exp_series)) if exp_series.notna().any() else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        target_salary = st.number_input("Mức lương mục tiêu (triệu VND/tháng)", min_value=0.0, max_value=max(1.0, sal_max), value=max(0.0, min(default_sal, sal_max)))
    with c2:
        target_exp = st.number_input("Số năm kinh nghiệm (ước lượng)", min_value=0.0, max_value=exp_max, value=max(0.0, min(default_exp, exp_max)))
    with c3:
        top_k = st.slider("Số ngành hot (Top K)", min_value=3, max_value=20, value=10, step=1)

    c4, c5 = st.columns(2)
    with c4:
        sigma_salary = st.slider("Độ rộng so khớp lương (σ)", min_value=1.0, max_value=max(2.0, (sal_max - sal_min) / 2 or 2.0), value=max(5.0, (sal_max - sal_min) / 10 or 5.0))
    with c5:
        sigma_exp = st.slider("Độ rộng so khớp kinh nghiệm (σ)", min_value=0.5, max_value=10.0, value=2.0)

    st.caption("Điểm 'hot' = exp(-((lương−mục tiêu)^2)/(2σ_l^2)) × exp(-((exp−mục tiêu)^2)/(2σ_e^2)).")

    def split_fields(s: str):
        s = (s or "").lower()
        s = s.replace(";", ",").replace("|", ",").replace("/", ",")
        parts = []
        for seg in s.split(","):
            parts.extend([x.strip() for x in seg.split(" - ")])
        parts = [p for p in parts if p and p != "nan"]
        return parts[:5]

    def stream_recommend_fields(frame: pd.DataFrame, sample_limit_per_field: int = 200):
        cols_needed = ["job_fields","salary_mid","experience_years","city","position_level","job_title"]
        cols_exist = [c for c in cols_needed if c in frame.columns]
        accum = {}
        two_sigma2_sal = 2.0 * (sigma_salary ** 2) + 1e-9
        two_sigma2_exp = 2.0 * (sigma_exp ** 2) + 1e-9

        for row in frame[cols_exist].itertuples(index=False, name=None):
            row_dict = dict(zip(cols_exist, row))
            s_mid = row_dict.get("salary_mid", np.nan)
            if pd.isna(s_mid): 
                continue
            s_mid = float(s_mid)

            ds = s_mid - float(target_salary)
            w_sal = np.exp(-(ds * ds) / two_sigma2_sal)

            e_years = row_dict.get("experience_years", np.nan)
            if pd.isna(e_years): 
                w_exp = 1.0
            else:
                de = float(e_years) - float(target_exp)
                w_exp = np.exp(-(de * de) / two_sigma2_exp)

            w = float(w_sal * w_exp)
            if w <= 0.0: 
                continue

            fields = split_fields(row_dict.get("job_fields", "")) or []
            for f in fields:
                b = accum.get(f)
                if b is None:
                    b = {"score": 0.0, "count": 0, "sum_salary": 0.0, "samples": []}
                    accum[f] = b
                b["score"] += w
                b["count"] += 1
                b["sum_salary"] += s_mid
                if len(b["samples"]) < sample_limit_per_field:
                    b["samples"].append({
                        "job_title": row_dict.get("job_title",""),
                        "city": row_dict.get("city",""),
                        "position_level": row_dict.get("position_level",""),
                        "salary_mid": s_mid,
                        "experience_years": e_years if not pd.isna(e_years) else None
                    })

        if not accum:
            return pd.DataFrame(columns=["field","score","count","avg_salary","med_salary","score_norm"]), {}

        fields, scores, counts, avg_salaries, med_salaries = [], [], [], [], []
        for f, b in accum.items():
            fields.append(f)
            scores.append(b["score"])
            counts.append(b["count"])
            avg_salaries.append(b["sum_salary"] / max(1, b["count"]))
            meds = [s["salary_mid"] for s in b["samples"] if s.get("salary_mid") is not None]
            med_salaries.append(float(np.median(meds)) if meds else np.nan)

        out = pd.DataFrame({
            "field": fields,
            "score": scores,
            "count": counts,
            "avg_salary": np.round(avg_salaries, 3),
            "med_salary": np.round(med_salaries, 3),
        })
        max_score = out["score"].max() if not out.empty else 0.0
        out["score_norm"] = (out["score"] / max_score).round(3) if max_score > 0 else 0.0
        return out, accum

    if st.button("🚀 Chạy dự đoán ngành hot"):
        base = filtered if len(filtered) > 0 else df
        results, accum = stream_recommend_fields(base)
        if results.empty:
            st.info("Chưa có dữ liệu phù hợp. Hãy nới lỏng bộ lọc hoặc tăng σ.")
        else:
            results = results.sort_values(by=["score","count","med_salary"], ascending=[False, False, False]).head(top_k)
            st.markdown("### Top ngành *hot*")
            st.dataframe(results.rename(columns={
                "field":"Ngành nghề",
                "score_norm":"Điểm (0-1)",
                "count":"Số tin",
                "avg_salary":"Lương TB",
                "med_salary":"Lương trung vị"
            }), width="stretch")
            st.bar_chart(results.set_index("field")["score_norm"])

            top_field = results.iloc[0]["field"]
            st.markdown(f"**Tin mẫu cho ngành đứng đầu:** _{top_field}_")
            samples = accum.get(top_field, {}).get("samples", [])[:10]
            st.dataframe(pd.DataFrame(samples), width="stretch") if samples else st.write("Không có mẫu hiển thị.")

# ====== TAB 3: Prophet ======
with tab_prophet:
    st.subheader("Dự báo nhu cầu tuyển dụng theo thời gian (Prophet)")
    date_col = None
    for c in filtered.columns:
        lc = c.lower()
        if any(k in lc for k in ["date","posted","time","created"]):
            date_col = c
            break

    if date_col:
        try:
            from prophet import Prophet
            ts = filtered[[date_col]].copy()
            ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
            ts = ts.dropna()
            if not ts.empty:
                ts["ds"] = ts[date_col].dt.tz_localize(None)
                ts["y"] = 1
                ts = ts.groupby("ds").agg(y=("y","sum")).reset_index()
                m = Prophet()
                m.fit(ts)
                future = m.make_future_dataframe(periods=90)
                fcst = m.predict(future)
                st.line_chart(fcst.set_index("ds")[["yhat"]])
                st.dataframe(fcst[["ds","yhat","yhat_lower","yhat_upper"]].tail(10), width="stretch")
            else:
                st.info("Không có dữ liệu ngày hợp lệ sau khi chuyển đổi.")
        except Exception as e:
            st.warning(f"Không thể chạy Prophet: {e}")
    else:
        st.info("Chưa phát hiện cột thời gian (ví dụ: posted_date). Hãy thêm cột để bật dự báo.")
