from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    BEST_MODEL_CONFUSION_MATRIX_FILE,
    CORRELATION_HEATMAP_FIGURE_FILE,
    DATA_QUALITY_HTML_FILE,
    FEATURE_IMPORTANCE_FIGURE_FILE,
    METRIC_COMPARISON_FIGURE_FILE,
    ROC_CURVE_FIGURE_FILE,
    SHAP_SUMMARY_FIGURE_FILE,
    TARGET_DISTRIBUTION_FIGURE_FILE,
    THRESHOLD_FIGURE_FILE,
)
from src.app_utils import (
    get_data_profile,
    get_feature_defaults,
    get_model_figure_path,
    get_saved_model_names,
    load_cv_results_table,
    load_data_quality_report,
    load_data_safe,
    load_model_safe,
    load_results_table,
    load_threshold_results_table,
    load_training_metadata,
    load_tuning_results_table,
    make_correlation_heatmap_figure,
    make_target_distribution_figure,
)
from src.batch_prediction import (
    create_prediction_template,
    export_prediction_results,
    find_unusual_values,
    load_demo_patients,
    load_prediction_file,
    predict_batch_dataframe,
    save_prediction_template,
    validate_prediction_batch,
)
from src.predict import predict_dataframe
from src.preprocessing import detect_target_column
from src.train import run_pipeline


MEDICAL_WARNING = (
    "Kết quả chỉ phục vụ học tập/demo, không thay thế tư vấn, xét nghiệm "
    "hoặc chẩn đoán của bác sĩ."
)


@st.cache_data(show_spinner=False)
def _cached_template_bytes(feature_names: tuple[str, ...]) -> bytes:
    return create_prediction_template(list(feature_names))


@st.cache_data(show_spinner=False)
def _cached_demo_patients(feature_names: tuple[str, ...]) -> pd.DataFrame:
    return load_demo_patients(list(feature_names))


def _format_metric_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    metric_columns = [
        column for column in df.columns
        if any(
            token in column
            for token in ["Accuracy", "Precision", "Recall", "F1", "Specificity", "MCC", "ROC-AUC", "PR-AUC"]
        )
    ]
    return df.style.format({column: "{:.4f}" for column in metric_columns})


def _feature_key(index: int, feature: str) -> str:
    safe_name = feature.replace(" ", "_").replace("/", "_").replace("-", "_")
    return f"manual_feature_{index}_{safe_name}"


def _single_input_dataframe(feature_names: list[str]) -> pd.DataFrame:
    values = {
        feature: st.session_state.get(_feature_key(index, feature), 0.0)
        for index, feature in enumerate(feature_names)
    }
    return pd.DataFrame([values], columns=feature_names)


def render_intro_page() -> None:
    st.title("Breast Cancer Classification")
    st.subheader("Machine Learning demo for Benign/Malignant classification")

    st.write(
        "Project xây dựng pipeline Machine Learning cho bài toán phân loại khối u vú "
        "thành **Benign** hoặc **Malignant**."
    )
    st.write(
        "Mục tiêu là demo quy trình có thể nộp báo cáo: kiểm định dữ liệu, tiền xử lý, "
        "huấn luyện, đánh giá, chọn model theo ưu tiên y tế và dự đoán thủ công/batch."
    )
    st.markdown(
        """
        **Pipeline tổng quan**

        1. Load dữ liệu từ `data/`.
        2. Kiểm tra schema, kiểu dữ liệu, missing values, duplicate, range, outlier, class balance.
        3. Tách train/test có `stratify=y`; scaler/imputer fit trên train set.
        4. Huấn luyện 6 model: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes.
        5. Đánh giá bằng Accuracy, Precision, Recall, F1-score, Specificity, ROC-AUC và confusion matrix.
        6. Chọn best model theo ưu tiên Recall lớp Malignant và giảm False Negative.
        7. Hỗ trợ single prediction, batch prediction, template Excel và demo patients.
        """
    )
    st.warning(MEDICAL_WARNING)


def render_data_page() -> None:
    st.title("Khám phá dữ liệu / EDA")
    df, error = load_data_safe()
    if error:
        st.error(error)
        st.stop()

    try:
        target_col = detect_target_column(df)
        profile = get_data_profile(df)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    rows = profile["shape"]["rows"]
    columns = profile["shape"]["columns"]
    feature_count = columns - 1

    col1, col2, col3 = st.columns(3)
    col1.metric("Số dòng", rows)
    col2.metric("Số cột", columns)
    col3.metric("Số feature", feature_count)

    st.subheader("Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Missing values")
    missing_df = pd.DataFrame(
        [{"column": column, "missing": count} for column, count in profile["missing_values"].items()]
    )
    st.dataframe(missing_df, use_container_width=True)

    st.subheader("Phân bố nhãn")
    label_df = pd.DataFrame(
        [{"label": label, "count": count} for label, count in profile["label_distribution"].items()]
    )
    st.dataframe(label_df, use_container_width=True)
    if TARGET_DISTRIBUTION_FIGURE_FILE.exists():
        st.image(str(TARGET_DISTRIBUTION_FIGURE_FILE))
    else:
        st.pyplot(make_target_distribution_figure(df, target_col), clear_figure=True)

    st.subheader("Heatmap correlation")
    if CORRELATION_HEATMAP_FIGURE_FILE.exists():
        st.image(str(CORRELATION_HEATMAP_FIGURE_FILE))
    else:
        heatmap = make_correlation_heatmap_figure(df)
        if heatmap is None:
            st.info("Không đủ cột số để vẽ heatmap correlation.")
        else:
            st.pyplot(heatmap, clear_figure=True)

    data_report = load_data_quality_report()
    if data_report:
        st.subheader("Data quality report")
        c1, c2, c3 = st.columns(3)
        c1.metric("Quality score", f"{data_report.get('quality_score', 'N/A')}/100")
        c2.metric("Duplicate rows", data_report.get("duplicate_rows", 0))
        c3.metric("Suspected outliers", data_report.get("outliers", {}).get("total_suspected_outliers", 0))

        st.write("Class balance:", data_report.get("class_balance", {}).get("counts", {}))
        high_corr = data_report.get("correlation", {}).get("highly_correlated_pairs", [])
        if high_corr:
            st.write("Feature tương quan cao |corr| > 0.95:")
            st.dataframe(pd.DataFrame(high_corr).head(20), use_container_width=True)
        if DATA_QUALITY_HTML_FILE.exists():
            st.caption(f"Báo cáo HTML đã lưu tại `{DATA_QUALITY_HTML_FILE}`.")


def render_training_page() -> None:
    st.title("Huấn luyện mô hình")
    _, error = load_data_safe()
    if error:
        st.error(error)
        return

    st.write("Nhấn nút bên dưới để chạy lại toàn bộ pipeline training và sinh báo cáo.")
    if st.button("Train model", type="primary"):
        with st.spinner("Đang huấn luyện mô hình, chạy CV/tuning và sinh báo cáo..."):
            try:
                result = run_pipeline()
                st.cache_resource.clear()
                st.cache_data.clear()
            except Exception as exc:
                st.error(f"Training thất bại: {exc}")
                return

        st.success("Training hoàn tất.")
        st.metric("Best model", result["best_model_name"])
        st.metric("Decision threshold", result.get("decision_threshold", "N/A"))
        st.write(result["selection_reason"])
        st.dataframe(result["results_df"], use_container_width=True)
        st.write("Model tốt nhất đã lưu tại `models/best_model.joblib`.")

    metadata = load_training_metadata()
    saved_models = get_saved_model_names()
    if metadata:
        st.subheader("Training gần nhất")
        st.write(f"Best model: **{metadata.get('best_model', 'N/A')}**")
        st.write(f"Selection rule: `{metadata.get('selection_rule', 'N/A')}`")
        st.write(f"Decision threshold: `{metadata.get('decision_threshold', 'N/A')}`")

    st.subheader("Model đã train")
    if saved_models:
        st.write(", ".join(saved_models))
    else:
        st.info("Chưa tìm thấy model đã train trong thư mục `models/`.")


def render_evaluation_page() -> None:
    st.title("Đánh giá mô hình")
    results_df = load_results_table()
    if results_df is None:
        st.warning("Chưa có bảng kết quả. Hãy train model trước.")
        return

    st.subheader("Bảng so sánh metric 6 model")
    st.dataframe(_format_metric_table(results_df), use_container_width=True)

    metadata = load_training_metadata()
    best_model = metadata.get("best_model") if metadata else results_df.index[0]
    best_row = results_df.loc[best_model]
    st.info(
        f"Best model hiện tại là **{best_model}**. Rule chọn model: ưu tiên Recall lớp "
        "Malignant, sau đó FN thấp, F1-score, ROC-AUC, Accuracy và cuối cùng là model "
        "đơn giản/dễ giải thích hơn."
    )
    st.write(
        f"Recall (M): **{best_row['Recall (M)']:.4f}**; "
        f"False Negative (FN): **{int(best_row['FN'])}**. "
        "FN là ca ác tính bị dự đoán nhầm thành lành tính, nên nguy hiểm hơn FP."
    )

    if METRIC_COMPARISON_FIGURE_FILE.exists():
        st.subheader("Biểu đồ so sánh metric")
        st.image(str(METRIC_COMPARISON_FIGURE_FILE))

    if BEST_MODEL_CONFUSION_MATRIX_FILE.exists():
        st.subheader("Confusion matrix của best model")
        st.image(str(BEST_MODEL_CONFUSION_MATRIX_FILE))

    if ROC_CURVE_FIGURE_FILE.exists():
        st.subheader("ROC curve tổng hợp")
        st.image(str(ROC_CURVE_FIGURE_FILE))

    threshold_df = load_threshold_results_table()
    if threshold_df is not None:
        st.subheader("Threshold optimization")
        st.dataframe(_format_metric_table(threshold_df), use_container_width=True)
        if THRESHOLD_FIGURE_FILE.exists():
            st.image(str(THRESHOLD_FIGURE_FILE))

    with st.expander("5-fold cross-validation"):
        cv_df = load_cv_results_table()
        if cv_df is None:
            st.info("Chưa có kết quả cross-validation.")
        else:
            st.dataframe(_format_metric_table(cv_df), use_container_width=True)

    with st.expander("Hyperparameter tuning"):
        tuning_df = load_tuning_results_table()
        if tuning_df is None:
            st.info("Chưa có kết quả tuning.")
        else:
            st.dataframe(tuning_df, use_container_width=True)

    with st.expander("Explainability"):
        if FEATURE_IMPORTANCE_FIGURE_FILE.exists():
            st.image(str(FEATURE_IMPORTANCE_FIGURE_FILE))
        if SHAP_SUMMARY_FIGURE_FILE.exists():
            st.image(str(SHAP_SUMMARY_FIGURE_FILE))
        if not FEATURE_IMPORTANCE_FIGURE_FILE.exists() and not SHAP_SUMMARY_FIGURE_FILE.exists():
            st.info("Chưa có hình explainability. Hãy kiểm tra `reports/model_explainability.md`.")

    model_name = st.selectbox("Chọn model để xem biểu đồ riêng", list(results_df.index))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion matrix")
        cm_path = get_model_figure_path(model_name, "confusion_matrix")
        if cm_path.exists():
            st.image(str(cm_path))
        else:
            st.warning("Chưa có confusion matrix cho model này.")

    with col2:
        st.subheader("ROC curve")
        roc_path = get_model_figure_path(model_name, "roc_curve")
        if roc_path.exists():
            st.image(str(roc_path))
        else:
            st.info("Model này chưa có ROC curve.")


def render_single_prediction(
    model,
    feature_names: list[str],
    label_mapping: dict[int, str],
    defaults: dict[str, float],
) -> None:
    st.subheader("Single Prediction")

    demo_df = _cached_demo_patients(tuple(feature_names))
    if not demo_df.empty:
        demo_name = st.selectbox("Demo Cases", ["Không dùng demo"] + demo_df["case_name"].tolist())
        if st.button("Load Demo Patient"):
            if demo_name != "Không dùng demo":
                demo_row = demo_df.loc[demo_df["case_name"] == demo_name].iloc[0]
                for index, feature in enumerate(feature_names):
                    st.session_state[_feature_key(index, feature)] = float(demo_row[feature])
                st.session_state["manual_review_ready"] = False
                st.rerun()
            st.warning("Hãy chọn một demo case trước khi load.")

    if st.button("Reset Form"):
        st.session_state["reset_form_requested"] = True

    if st.session_state.get("reset_form_requested"):
        st.warning("Bạn có chắc muốn reset toàn bộ form về giá trị mặc định?")
        col_confirm, col_cancel = st.columns(2)
        if col_confirm.button("Confirm Reset"):
            for index, feature in enumerate(feature_names):
                st.session_state[_feature_key(index, feature)] = defaults[feature]
            st.session_state["manual_review_ready"] = False
            st.session_state["reset_form_requested"] = False
            st.rerun()
        if col_cancel.button("Cancel Reset"):
            st.session_state["reset_form_requested"] = False
            st.rerun()

    columns = st.columns(3)
    for index, feature in enumerate(feature_names):
        key = _feature_key(index, feature)
        if key not in st.session_state:
            st.session_state[key] = defaults[feature]
        with columns[index % 3]:
            st.number_input(feature, key=key, format="%.6f")

    input_df = _single_input_dataframe(feature_names)
    unusual_values = find_unusual_values(input_df, feature_names)
    if unusual_values:
        st.warning(
            "Medical Validation Warning: Một số giá trị nằm ngoài khoảng thông thường. "
            "Vui lòng kiểm tra lại trước khi dự đoán."
        )
        st.dataframe(pd.DataFrame(unusual_values), use_container_width=True)

    if st.button("Review Input", type="primary"):
        st.session_state["manual_review_ready"] = True

    if st.session_state.get("manual_review_ready"):
        st.subheader("Review")
        review_df = input_df.T.reset_index()
        review_df.columns = ["Feature", "Value"]
        st.dataframe(review_df, use_container_width=True)

        col_back, col_predict = st.columns(2)
        if col_back.button("Back To Edit"):
            st.session_state["manual_review_ready"] = False
            st.rerun()
        if col_predict.button("Confirm And Predict", type="primary"):
            try:
                result = predict_dataframe(input_df, model, feature_names, label_mapping)
            except Exception as exc:
                st.error(f"Dự đoán thất bại: {exc}")
                return

            prediction = str(result["prediction"]).casefold()
            display_label = "Malignant" if prediction == "malignant" or result["prediction_code"] == 1 else "Benign"
            st.metric("Kết quả dự đoán", display_label)
            st.write(f"Decision threshold đang dùng: **{result.get('decision_threshold', 0.5):.2f}**")
            if "probability_benign" in result and "probability_malignant" in result:
                st.write(f"Xác suất Benign: **{result['probability_benign'] * 100:.2f}%**")
                st.write(f"Xác suất Malignant: **{result['probability_malignant'] * 100:.2f}%**")


def render_batch_prediction(
    model,
    feature_names: list[str],
    label_mapping: dict[int, str],
) -> None:
    st.subheader("Batch Prediction")

    save_prediction_template(feature_names)
    template_bytes = _cached_template_bytes(tuple(feature_names))
    st.download_button(
        "Download Prediction Template",
        data=template_bytes,
        file_name="breast_cancer_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    uploaded_file = st.file_uploader("Upload CSV hoặc XLSX", type=["csv", "xlsx"])
    if uploaded_file is None:
        return

    try:
        batch_df = load_prediction_file(uploaded_file)
    except Exception as exc:
        st.error(f"Không đọc được file upload: {exc}")
        return

    st.write("Records Loaded:", len(batch_df))
    st.dataframe(batch_df.head(20), use_container_width=True)

    try:
        validated_df, report = validate_prediction_batch(batch_df, feature_names)
    except Exception as exc:
        st.error(f"Validation thất bại: {exc}")
        return

    st.subheader("Validation Report")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records Loaded", report["records_loaded"])
    col2.metric("Missing Values", sum(item["missing_count"] for item in report["missing_values"]))
    col3.metric("Duplicate Rows", report["duplicate_rows"]["duplicate_rows"])
    col4.metric("Outliers Detected", report["outliers"]["total_suspected_outliers"])

    if report["extra_columns"]:
        st.info(f"Cột ngoài schema sẽ được bỏ qua khi predict: {', '.join(report['extra_columns'])}")

    invalid_df = pd.DataFrame(report["invalid_cells"])
    if not invalid_df.empty:
        st.error("File upload còn lỗi cần sửa trước khi predict.")
        st.dataframe(invalid_df, use_container_width=True)

    unusual_df = pd.DataFrame(report["unusual_values"])
    if not unusual_df.empty:
        st.warning("Medical Validation Warning: Có giá trị nằm ngoài khoảng thông thường.")
        st.dataframe(unusual_df, use_container_width=True)

    if not report["can_predict"]:
        return

    if st.button("Confirm And Predict Batch", type="primary"):
        try:
            results_df = predict_batch_dataframe(validated_df, model, feature_names, label_mapping)
        except Exception as exc:
            st.error(f"Batch prediction thất bại: {exc}")
            return

        st.success("Batch prediction hoàn tất.")
        st.dataframe(results_df, use_container_width=True)
        st.download_button(
            "Download prediction_results.xlsx",
            data=export_prediction_results(results_df),
            file_name="prediction_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def render_prediction_page() -> None:
    st.title("Dự đoán")
    st.warning(MEDICAL_WARNING)

    assets, model_error = load_model_safe()
    if model_error:
        st.error(model_error)
        return

    model, feature_names, label_mapping = assets
    df, data_error = load_data_safe()
    if data_error:
        st.warning(
            "Không load được dataset nên giá trị mặc định sẽ là 0. "
            "Bạn vẫn có thể nhập thủ công."
        )
        df = None

    defaults = get_feature_defaults(df, feature_names)
    tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])
    with tab_single:
        render_single_prediction(model, feature_names, label_mapping, defaults)
    with tab_batch:
        render_batch_prediction(model, feature_names, label_mapping)


PAGES = {
    "Giới thiệu": render_intro_page,
    "Khám phá dữ liệu": render_data_page,
    "Huấn luyện mô hình": render_training_page,
    "Đánh giá mô hình": render_evaluation_page,
    "Dự đoán": render_prediction_page,
}
