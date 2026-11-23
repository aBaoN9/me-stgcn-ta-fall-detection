import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_streamlit.data import BASELINE_NAMES, MODEL_CHOICES, MODEL_NAMES, RUNS, SEEDS, load_metrics
from demo_streamlit.history import HistoryStore, history_label
from demo_streamlit.jobs import AnalysisJob
from demo_streamlit.player import player_height, player_html
from demo_streamlit.prepare_assets import verify_pose_asset
from demo_streamlit.reporting import readable_report, report_data, summary, trace_csv
from demo_streamlit.video import MAX_BYTES, PIPELINE_VERSION


st.set_page_config(page_title="FallScope | Video Lab", page_icon="FS", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
html,body,.stApp,p,button,input,select,label,h1,h2,h3{font-family:'Be Vietnam Pro','Trebuchet MS',sans-serif}
.stApp{background:radial-gradient(ellipse at 95% 0%,#e3eeff,transparent 55%),#f5f8fe;color:#183354}
.block-container{max-width:1450px;padding:1.5rem 2.5rem 3rem}
[data-testid="stHeader"]{display:none}
h1,h2,h3{color:#183354;letter-spacing:-.035em}
.app-brand{font-size:20px;font-weight:700;letter-spacing:-.04em;color:#215bd8;margin-bottom:20px}
[data-testid="stFileUploader"]{border-radius:12px}
[data-testid="stMetric"]{background:#fff;border:1px solid #dce6f4;border-radius:12px;padding:16px}
[data-testid="stMetricValue"]{color:#215bd8;font-variant-numeric:tabular-nums}
.stButton>button{border-radius:8px}
@media(max-width:760px){.block-container{padding:1.2rem 1rem}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def metrics(run_key):
    return load_metrics(run_key)


def render_report(result, model_name, prefix):
    totals = summary(result)
    columns = st.columns(4)
    columns[0].metric("Vùng cần xem lại", totals["events"])
    columns[1].metric("Thời gian score cao", f'{totals["high_seconds"]:.2f}s')
    columns[2].metric("Cửa sổ hợp lệ", f'{totals["valid_windows"]}/{totals["total_windows"]}')
    columns[3].metric("Thời gian xử lý", f'{totals["processing_seconds"]:.1f}s')
    st.caption(f'Khởi động: {totals["warmup_seconds"]:.2f}s · Thiếu dữ liệu sau khởi động: '
               f'{totals["unavailable_seconds"]:.2f}s. Vùng score cao chưa phải ngã đã được xác nhận.')
    downloads = st.columns(3)
    downloads[0].download_button("Báo cáo HTML / in PDF", readable_report(result, model_name),
                                 file_name="fallscope-report.html", mime="text/html", key=f"{prefix}_html", on_click="ignore")
    downloads[1].download_button("Trace CSV", trace_csv(result), file_name="fallscope-trace.csv",
                                 mime="text/csv", key=f"{prefix}_csv", on_click="ignore")
    downloads[2].download_button("Metadata JSON", json.dumps(report_data(result), ensure_ascii=False,
                                                            indent=2, allow_nan=False),
                                 file_name="fallscope-analysis.json", mime="application/json", key=f"{prefix}_json", on_click="ignore")


st.markdown('<div class="app-brand">FallScope</div>', unsafe_allow_html=True)

run_key = "accepted"
run = RUNS[run_key]

upload_column, model_column = st.columns([1.85, 1], gap="large")
with upload_column:
    uploaded = st.file_uploader("Video đầu vào", type=["mp4", "mov", "avi", "webm"],
                                key=f"upload_{st.session_state.get('upload_generation', 0)}",
                                help="Tối đa 50 MB, 60 giây, Full HD và 1–60 FPS. Video một người; preview không âm thanh.")
    st.caption("Xử lý trên máy này · tối đa 60 giây / 50 MB")
with model_column:
    variant = st.selectbox("Mô hình · Biến thể", tuple(MODEL_CHOICES),
                           index=tuple(MODEL_CHOICES).index("A3_motion_attention"),
                           format_func=lambda key: MODEL_CHOICES[key], key="variant")

available = variant not in BASELINE_NAMES
if not available:
    st.info("Baseline có notebook lịch sử nhưng chưa có checkpoint đã kiểm chứng trong bộ demo. "
            "Chưa thể phân tích video với lựa chọn này; không tự train hoặc dùng weights của model khác.")

with st.expander("Cài đặt nâng cao"):
    settings = st.columns(3)
    seed = settings[0].selectbox("Seed", SEEDS, key="seed")
    subject = settings[1].selectbox("Fold giữ lại subject", (1, 2, 3, 4), key="subject")
    window_seconds = settings[2].slider("Cửa sổ quá khứ (giây)", 1.0, 6.0, 3.0, .5, key="window_seconds")
    st.caption("Mỗi lựa chọn dùng một checkpoint fold nghiên cứu, không phải ensemble hoặc model deployment. Video tải lên không có bảo đảm held-out. Cửa sổ cần ít nhất 32 frame; chỉ nội suy trong quá khứ, tối thiểu 50% frame có pose và frame cuối có pose.")
    st.caption("MediaPipe full v1 → 33 landmarks → feature v2 (8 × 32 × 33) → model CPU → score. Ngưỡng 0.50 cố định; cập nhật khoảng 0.25 giây video. Đây là adaptation theo cửa sổ, chưa có benchmark online.")

video_bytes = uploaded.getvalue() if uploaded is not None else None
video_hash = hashlib.sha256(video_bytes).hexdigest() if video_bytes else None
signature = (video_hash, run_key, variant, seed, subject, window_seconds, PIPELINE_VERSION)
if st.session_state.get("analysis_signature") != signature:
    if st.session_state.get("job") is not None:
        st.session_state["job"].cancel()
    st.session_state.pop("analysis", None)
    st.session_state.pop("saved_current", None)
    st.session_state["analysis_signature"] = signature

asset_error = None
try:
    verify_pose_asset()
except (ValueError, FileNotFoundError) as exc:
    asset_error = str(exc)
if asset_error:
    st.warning("Chưa có pose asset đúng hash. Chạy python -m demo_streamlit.prepare_assets trước khi phân tích.")
oversized = video_bytes is not None and len(video_bytes) > MAX_BYTES
if oversized:
    st.error("Video vượt giới hạn 50 MB. Hãy chọn video ngắn hơn.")

actions = st.columns([1.85, 1], gap="large")
job = st.session_state.get("job")
busy = job is not None and job.snapshot()[0] == "running"
analyze = actions[0].button("Phân tích video", type="primary", width="stretch",
                           disabled=not video_bytes or not available or busy or bool(asset_error) or oversized,
                           key="analyze")
reset = actions[1].button("Xóa video / kết quả", width="stretch", key="reset")
if reset:
    if job is not None:
        job.cancel()
    st.session_state.pop("analysis", None)
    st.session_state.pop("saved_current", None)
    st.session_state["upload_generation"] = st.session_state.get("upload_generation", 0) + 1
    st.rerun()
if analyze and not busy:
    st.session_state.pop("analysis", None)
    st.session_state.pop("saved_current", None)
    st.session_state["job"] = AnalysisJob(video_bytes, signature, uploaded.name)
    st.rerun()


@st.fragment(run_every=.5 if st.session_state.get("job") is not None else None)
def job_progress():
    active_job = st.session_state.get("job")
    if active_job is None:
        return
    status, fraction, message = active_job.snapshot()
    if status == "running":
        st.progress(fraction, text=message)
        if st.button("Hủy phân tích", key="cancel_job"):
            active_job.cancel()
            st.info("Đang dừng xử lý và dọn file tạm. Có thể tiếp tục chọn video/mô hình.")
    else:
        output = active_job.take_result()
        if status == "completed" and active_job.signature == st.session_state["analysis_signature"] and output:
            st.session_state["analysis"] = output
        elif status in ("failed", "cancelled"):
            st.session_state["job_notice"] = (status, message)
        st.session_state.pop("job", None)
        st.rerun()


job_progress()
notice = st.session_state.pop("job_notice", None)
if notice:
    (st.error if notice[0] == "failed" else st.info)(notice[1])

analysis = st.session_state.get("analysis")
preview, result = analysis if analysis else (None, None)
if uploaded is not None and result is None and available and st.session_state.get("job") is None:
    st.info("Video đã được chọn. Nhấn Phân tích video; chưa có điểm số hoặc kết luận.")
if result is not None and not result["valid_windows"]:
    st.warning("Không có cửa sổ đủ dữ liệu: video quá ngắn hoặc thiếu pose. Không kết luận ADL/Fall.")
components.html(player_html(preview, result, MODEL_CHOICES[variant]), height=player_height(result), scrolling=True)

if result is not None:
    st.subheader("Tóm tắt phiên phân tích")
    render_report(result, MODEL_CHOICES[variant], "current")
    with st.expander("Nguồn mô hình và chi tiết phiên"):
        st.json({"checkpoint": result["identity"], "pose_asset_sha256": result["pose_asset_sha256"],
                 "video_sha256": result["video_sha256"], "pipeline": result["pipeline"],
                 "pose_seconds": result["pose_seconds"], "processing_seconds": result["processing_seconds"]})
        st.caption("File báo cáo không chứa video, ảnh hoặc landmarks. Lưu lịch sử là thao tác riêng, không tự động.")
    with st.expander("Lưu phiên vào lịch sử"):
        include_video = st.checkbox("Lưu kèm video không âm thanh, ảnh sự kiện và skeleton để phát lại",
                                    key="include_video", value=False)
        st.caption("Mặc định chỉ lưu báo cáo và tên video, không lưu hình ảnh/landmarks. Nếu chọn lưu kèm video, "
                   "người có quyền truy cập máy này có thể xem lại. Dữ liệu không mã hóa, tối đa 20 phiên / 250 MB; "
                   "không đưa vào Git hoặc gửi dịch vụ ngoài. Xóa ở Lịch sử để bỏ bản đã lưu.")
        if st.button("Lưu phiên trên máy này", key="save_current", disabled=bool(st.session_state.get("saved_current"))):
            try:
                st.session_state["saved_current"] = HistoryStore().save(
                    result, MODEL_CHOICES[variant], preview, include_video)
                st.success("Đã lưu phiên. Mở mục Lịch sử để xem lại.")
            except (ValueError, OSError, sqlite3.Error) as exc:
                st.error(f"Không lưu được phiên: {exc}")

with st.expander("Lịch sử trên máy này", expanded=False):
    st.caption("Chỉ có các phiên bạn chủ động lưu. Đây là lịch sử local dùng chung cho người truy cập ứng dụng "
               "trên máy này, chưa có tài khoản/phân quyền. Không nên mở cổng public.")
    store = HistoryStore()
    try:
        entries = store.entries()
        if not entries:
            st.info("Chưa có phiên đã lưu.")
        else:
            inventory = {entry["id"]: entry for entry in entries}
            selected_history = st.selectbox("Phiên đã lưu", tuple(inventory), key="history_selection",
                                            format_func=lambda key: history_label(inventory[key]))
            history_actions = st.columns(2)
            if history_actions[0].button("Mở phiên đã lưu", key="open_history"):
                st.session_state["history_view"] = (selected_history, store.load(selected_history))
            delete_confirmed = st.checkbox("Tôi muốn xóa vĩnh viễn phiên đã chọn", key=f"delete_{selected_history}")
            if history_actions[1].button("Xóa phiên đã chọn", key="delete_history", disabled=not delete_confirmed):
                store.delete(selected_history)
                st.session_state.pop("history_view", None)
                st.session_state.pop("saved_current", None)
                st.rerun()
            opened = st.session_state.get("history_view")
            if opened and opened[0] == selected_history:
                saved_preview, saved_result, saved_model = opened[1]
                st.write(f"Phiên đã lưu · {saved_model}")
                if saved_preview:
                    components.html(player_html(saved_preview, saved_result, saved_model),
                                    height=player_height(saved_result), scrolling=True)
                else:
                    st.info("Phiên này chỉ lưu báo cáo; không có video, ảnh hoặc skeleton để phát lại.")
                    st.dataframe(report_data(saved_result)["events"], width="stretch", column_config={
                        "start": st.column_config.NumberColumn("Bắt đầu (s)", format="%.2f"),
                        "end": st.column_config.NumberColumn("Kết thúc (s)", format="%.2f"),
                        "peak": st.column_config.NumberColumn("Score cao nhất", format="%.3f")})
                render_report(saved_result, saved_model, "history")
    except (ValueError, OSError, sqlite3.Error) as exc:
        st.error(f"Không đọc được lịch sử: {exc}")

with st.expander("Benchmark đã lưu — không phải chỉ số live của video", expanded=False):
    if not available:
        st.info("Baseline nằm trong notebook 02/03, khác protocol với benchmark ba seed hiện tại. "
                "Chưa đưa số lịch sử vào cùng bảng so sánh hoặc gán chỉ số của A0/A3 cho ST-GCN base.")
    try:
        table = metrics(run_key)
        selected = table.loc[table.variant == variant]
        labels = {"accuracy": "Accuracy", "f1": "F1 / Fall", "sensitivity": "Sensitivity", "specificity": "Specificity"}
        if available:
            for column, (metric, label) in zip(st.columns(4), labels.items()):
                column.metric(label, f"{selected[metric].mean()*100:.2f}%", f"SD {selected[metric].std(ddof=1)*100:.2f} pp", delta_color="off")
        st.caption("Mean ± sample SD qua 3 seed; cùng 160 clip / 4 subject, không phải 480 video độc lập. Không tính Accuracy/F1 mới từ video chưa có ground truth. Các chỉ số này không đổi theo timestamp.")
        comparison = table.groupby("variant")[["accuracy", "f1", "sensitivity", "specificity"]].mean().rename(index=MODEL_NAMES)
        st.dataframe((comparison * 100).round(2), width="stretch")
        st.caption(run.provenance)
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"Không đọc được bằng chứng benchmark: {exc}")

st.caption("RESEARCH ONLY / Video một người · không phải hệ thống cấp cứu, giám sát an toàn hoặc chẩn đoán y tế · không thay model/threshold deployment.")
