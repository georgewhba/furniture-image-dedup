"""Streamlit Web Application for Furniture Deduplication System.
نظام ويب تفاعلي ذكي لفرز وكشف صور الأثاث المكررة والمتطابقة.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from PIL import Image
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, load_config
from src.pipeline import run as run_pipeline
from src.scoring import DuplicateGroup

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Page Configuration & Modern Arabic Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="نظام فرز وتطابق صور الأثاث | Furniture Deduplication AI",
    page_icon="🪑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Cairo font, RTL direction, and polished design tokens
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Cairo', -apple-system, BlinkMacSystemFont, sans-serif !important;
        letter-spacing: normal !important;
    }

    /* RTL base */
    .rtl-text {
        direction: rtl;
        text-align: right;
    }

    /* Main header banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #0d9488 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        direction: rtl;
        text-align: right;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #ffffff;
        line-height: 1.3;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        color: #e0f2fe;
        line-height: 1.6;
    }
    .hero-badges {
        margin-top: 1rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    .badge-confidence-high {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-confidence-med {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-confidence-low {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🪑 نظام كشف وتصفية صور الأثاث المتطابقة بالذكاء الاصطناعي</div>
        <div class="hero-subtitle">
            فرز ذكي فائق الدقة لكتالوجات ومكتبات الأثاث والتجارة الإلكترونية — يكتشف نفس المنتج بنفس اللون بدقة، ويستبعد تلقائياً المنتجات ذات الأشكال المتشابهة والألوان المختلفة، مع الحفاظ الكامل على صورك دون أي حذف تلقائي.
        </div>
        <div class="hero-badges">
            <span class="hero-badge">🔒 أمان 100% (Read-Only)</span>
            <span class="hero-badge">🎨 بوابة فحص الألوان (CIE-Lab & rembg)</span>
            <span class="hero-badge">⚡ فهرس FAISS وسرعة فائقة</span>
            <span class="hero-badge">📊 تقارير Excel و CSV تفصيلية</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar: Settings and Controls
# ---------------------------------------------------------------------------

st.sidebar.markdown("<h2 class='rtl-text'>⚙️ إعدادات وخيارات الفرز</h2>", unsafe_allow_html=True)

# 1. Processing Scope
st.sidebar.markdown("### 🎯 نطاق المعالجة")
sample_mode = st.sidebar.selectbox(
    "حجم البيانات المفحوصة:",
    options=["كافة الصور (Full Dataset)", "عينة تجريبية أولى (500 صورة)", "عينة تجريبية أولى (1000 صورة)", "عينة سريعة (100 صورة)", "رقم مخصص"],
    index=0,
)

sample_size: int | None = None
if sample_mode == "عينة سريعة (100 صورة)":
    sample_size = 100
elif sample_mode == "عينة تجريبية أولى (500 صورة)":
    sample_size = 500
elif sample_mode == "عينة تجريبية أولى (1000 صورة)":
    sample_size = 1000
elif sample_mode == "رقم مخصص":
    sample_size = st.sidebar.number_input("حدد عدد الصور:", min_value=10, max_value=100000, value=500, step=50)

# 2. Advanced Color & AI Settings
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 بوابة التحقق من الألوان (Color Gate)")
st.sidebar.caption("الشرط الجوهري: منع تصنيف كنب أو كرسي بلون مختلف كتكرار، حتى لو كان نفس الشكل والموديل.")

use_bg_removal = st.sidebar.checkbox(
    "عزل خلفية المنتج بالذكاء الاصطناعي (rembg)",
    value=True,
    help="يقوم بعزل الأثاث عن الخلفية (الجدار، الغرفة) لمقارنة لون المنتج الفعلي فقط ومنع تلوث الألوان.",
)

color_distance_threshold = st.sidebar.slider(
    "عتبة المسافة اللونية (Delta-E):",
    min_value=10.0,
    max_value=40.0,
    value=25.0,
    step=1.0,
    help="المسافة في فضاء CIE-Lab اللوني. القيمة الأقل (مثل 20) تكون أكثر صرامة في اشتراط تطابق اللون التام.",
)

similarity_threshold = st.sidebar.slider(
    "عتبة تشابه الموديل والشكل (CLIP Cosine):",
    min_value=0.70,
    max_value=0.95,
    value=0.85,
    step=0.01,
    help="درجة تشابه هيكل وتصميم المنتج بزوايا مختلفة.",
)

phash_threshold = st.sidebar.slider(
    "عتبة الهاش الإدراكي (pHash Distance):",
    min_value=4,
    max_value=16,
    value=10,
    step=1,
    help="لكشف الصور المتطابقة بعد القص أو الضغط أو إضافة علامة مائية.",
)

st.sidebar.markdown("---")
enable_resume = st.sidebar.checkbox("تفعيل الكاش والاستئناف السريع (Cache)", value=True)


# ---------------------------------------------------------------------------
# Helper: Count images
# ---------------------------------------------------------------------------

def count_supported_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


def extract_zip(zip_bytes_or_path: io.BytesIO | Path, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_bytes_or_path, "r") as zf:
        zf.extractall(dest_dir)
    return count_supported_images(dest_dir)


# ---------------------------------------------------------------------------
# Data Input Modes (4 Comprehensive Tabs)
# ---------------------------------------------------------------------------

tab_upload, tab_gdrive, tab_kaggle, tab_folder = st.tabs([
    "📦 رفع ملف مضغوط ZIP (Upload)",
    "🌐 رابط Google Drive",
    "📊 داتاست Kaggle",
    "📁 مجلد محلي على الجهاز (Local Folder)",
])

# 1. TAB: ZIP Upload
with tab_upload:
    st.markdown("<p class='rtl-text'><b>رفع ملف مضغوط ZIP يحتوي على الصور:</b></p>", unsafe_allow_html=True)
    uploaded_zip = st.file_uploader(
        "اختر ملف مضغوط ZIP للصور:",
        type=["zip"],
        help="ارفع ملف مضغوط يحتوي على مجلد الصور.",
        key="uploader_zip",
    )

    if uploaded_zip is not None:
        if "last_uploaded_name" not in st.session_state or st.session_state["last_uploaded_name"] != uploaded_zip.name:
            temp_dir = Path(tempfile.mkdtemp(prefix="furniture_upload_"))
            with st.spinner("جاري فك ضغط ملف الصور..."):
                unzipped_count = extract_zip(io.BytesIO(uploaded_zip.read()), temp_dir)
            st.session_state["target_image_dir"] = str(temp_dir)
            st.session_state["last_uploaded_name"] = uploaded_zip.name
            st.session_state["image_count"] = unzipped_count
            st.success(f"✅ تم فك الضغط بنجاح! تم استخراج **{unzipped_count:,}** صورة.")


# 2. TAB: Google Drive Link
with tab_gdrive:
    st.markdown("<p class='rtl-text'><b>تنزيل الصور مباشرة من رابط Google Drive:</b></p>", unsafe_allow_html=True)
    st.caption("تأكد أن إذن مشاركة الملف في Google Drive هو: **أي شخص لديه الرابط (Anyone with the link)**.")
    
    gdrive_url = st.text_input(
        "رابط ملف Google Drive (ZIP أو مجلد):",
        placeholder="https://drive.google.com/file/d/1A2B3C.../view?usp=sharing",
        key="gdrive_input",
    )

    if st.button("📥 تنزيل وفك الضغط من Google Drive", key="btn_gdrive"):
        if not gdrive_url.strip():
            st.warning("⚠️ يرجى إدخال رابط Google Drive أولاً.")
        else:
            try:
                import gdown
                temp_dir = Path(tempfile.mkdtemp(prefix="gdrive_download_"))
                with st.spinner("جاري تنزيل الملفات من Google Drive بسرعة السيرفر... قد يستغرق لحظات حسب الحجم:"):
                    # Check if it's a folder URL
                    if "drive/folders" in gdrive_url:
                        gdown.download_folder(url=gdrive_url, output=str(temp_dir), quiet=False)
                    else:
                        zip_target = temp_dir / "dataset.zip"
                        downloaded = gdown.download(url=gdrive_url, output=str(zip_target), quiet=False, fuzzy=True)
                        if downloaded and zip_target.exists():
                            try:
                                extract_zip(zip_target, temp_dir)
                                zip_target.unlink(missing_ok=True)
                            except zipfile.BadZipFile:
                                pass
                
                n_imgs = count_supported_images(temp_dir)
                if n_imgs > 0:
                    st.session_state["target_image_dir"] = str(temp_dir)
                    st.session_state["image_count"] = n_imgs
                    st.success(f"🎉 تم تنزيل واستخراج **{n_imgs:,}** صورة من Google Drive بنجاح!")
                else:
                    st.error("❌ تم التنزيل ولكن لم يتم العثور على صور مدعومة، يرجى التأكد من الرابط ومحتوى الملف.")
            except Exception as exc:
                st.error(f"❌ حدث خطأ أثناء التنزيل من Google Drive: {exc}")


# 3. TAB: Kaggle Dataset Link
with tab_kaggle:
    st.markdown("<p class='rtl-text'><b>تنزيل الصور مباشرة من Kaggle:</b></p>", unsafe_allow_html=True)
    st.caption("أدخل اسم أو رابط داتاست كاجل (مثال: `owner/dataset-name`).")
    
    kaggle_slug_input = st.text_input(
        "معرّف داتاست كاجل (Dataset Slug / URL):",
        placeholder="مثال: rhtsingh/google-universal-image-embeddings-128x128",
        key="kaggle_input",
    )

    k_col1, k_col2 = st.columns(2)
    with k_col1:
        k_user = st.text_input("Kaggle Username (اختياري إن تم إعداده مسبقاً):", type="default")
    with k_col2:
        k_key = st.text_input("Kaggle Key / Token (اختياري):", type="password")

    if st.button("📥 تنزيل وفك ضغط الداتاست من Kaggle", key="btn_kaggle"):
        if not kaggle_slug_input.strip():
            st.warning("⚠️ يرجى إدخال اسم الداتاست في Kaggle.")
        else:
            # Set credentials if provided
            if k_user.strip():
                os.environ["KAGGLE_USERNAME"] = k_user.strip()
            if k_key.strip():
                os.environ["KAGGLE_KEY"] = k_key.strip()
                
            # Extract slug if user pasted full URL
            slug = kaggle_slug_input.strip()
            match = re.search(r"kaggle\.com/(?:datasets/)?([^/]+/[^/?]+)", slug)
            if match:
                slug = match.group(1)

            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                
                temp_dir = Path(tempfile.mkdtemp(prefix="kaggle_dataset_"))
                with st.spinner(f"جاري تنزيل الداتاست {slug} من Kaggle..."):
                    api.dataset_download_files(slug, path=str(temp_dir), unzip=True, quiet=False)
                    
                n_imgs = count_supported_images(temp_dir)
                if n_imgs > 0:
                    st.session_state["target_image_dir"] = str(temp_dir)
                    st.session_state["image_count"] = n_imgs
                    st.success(f"🎉 تم تنزيل واستخراج **{n_imgs:,}** صورة من Kaggle بنجاح!")
                else:
                    st.error("❌ تم التنزيل ولكن لم يتم العثور على صور مدعومة داخل الداتاست.")
            except Exception as exc:
                st.error(f"❌ خطأ أثناء الاتصال بـ Kaggle: {exc}\nتأكد من إدخال الـ Username والـ Key الصحيحين في حسابك على Kaggle.")


# 4. TAB: Local Folder (For PC / Server execution)
with tab_folder:
    st.markdown("<p class='rtl-text'><b>أدخل مسار مجلد موجود محلياً على جهازك أو السيرفر:</b></p>", unsafe_allow_html=True)
    folder_input = st.text_input(
        "مسار مجلد الصور المحلي:",
        placeholder=r"مثال: D:\furniture_catalog\images",
        key="local_folder_input",
    )

    if folder_input.strip():
        resolved_folder = Path(folder_input.strip().strip('"').strip("'"))
        if resolved_folder.exists() and resolved_folder.is_dir():
            found_count = count_supported_images(resolved_folder)
            st.session_state["target_image_dir"] = str(resolved_folder)
            st.session_state["image_count"] = found_count
            st.success(f"✅ تم العثور على المجلد بنجاح! يحتوي على **{found_count:,}** صورة مدعومة.")
        else:
            st.error("❌ المسار غير موجود أو ليس مجلداً صحيحاً، يرجى التأكد من المسار.")


# ---------------------------------------------------------------------------
# Active Dataset Status Banner
# ---------------------------------------------------------------------------

active_dir_str = st.session_state.get("target_image_dir")
active_count = st.session_state.get("image_count", 0)

st.markdown("<br>", unsafe_allow_html=True)
if active_dir_str and Path(active_dir_str).exists():
    st.info(f"📂 **المجلد الجاهز للفرز حالياً:** `{active_dir_str}` — يحتوي على **{active_count:,}** صورة جاهزة للتحليل.")
else:
    st.warning("👈 يرجى رفع ملف ZIP أو وضع رابط Google Drive / Kaggle أو اختيار مجلد من التبويبات أعلاه للمتابعة.")


# ---------------------------------------------------------------------------
# Execution Section
# ---------------------------------------------------------------------------

run_col1, run_col2, run_col3 = st.columns([1, 2, 1])

with run_col2:
    start_clicked = st.button(
        "🚀 بدء الفرز والتحليل الذكي الآن",
        type="primary",
        use_container_width=True,
        disabled=(active_dir_str is None),
    )

if start_clicked and active_dir_str is not None:
    target_path = Path(active_dir_str)
    
    # 1. Load base config safely
    base_config_path = PROJECT_ROOT / "config.yaml"
    base_cfg = load_config(base_config_path) if base_config_path.exists() else Config()

    # 2. Immutable replace to avoid FrozenInstanceError
    config = replace(
        base_cfg,
        use_background_removal=use_bg_removal,
        color_distance_threshold=float(color_distance_threshold),
        similarity_threshold=float(similarity_threshold),
        phash_threshold=int(phash_threshold),
    )

    # Output paths
    run_timestamp = int(time.time())
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_xlsx = output_dir / f"furniture_dedup_report_{run_timestamp}.xlsx"
    cache_dir = PROJECT_ROOT / ".dedup_cache"

    progress_box = st.container()
    with progress_box:
        st.markdown("<h3 class='rtl-text'>⏳ جاري تنفيذ مراحل الفرز الست...</h3>", unsafe_allow_html=True)
        status_text = st.empty()
        prog_bar = st.progress(0.1)

        status_text.info("🔍 جاري تنفيذ المراحل: الهاش الدقيق + الهاش الإدراكي + تمثيلات CLIP + فحص الألوان (CIE-Lab)...")
        prog_bar.progress(0.3)

        t_start = time.time()
        try:
            with st.spinner("جاري استخراج المتجهات البصرية وفحص الألوان..."):
                groups = run_pipeline(
                    input_dir=target_path,
                    output_path=output_xlsx,
                    config=config,
                    cache_dir=cache_dir,
                    sample_size=sample_size,
                    resume=enable_resume,
                    verbose=False,
                )

            prog_bar.progress(1.0)
            elapsed_time = time.time() - t_start
            status_text.success(f"🎉 اكتمل الفرز بنجاح في غضون {elapsed_time:.1f} ثانية!")
            
            # Save results in session state
            st.session_state["results_groups"] = groups
            st.session_state["output_xlsx"] = str(output_xlsx)
            st.session_state["output_csv"] = str(output_xlsx.with_suffix(".csv"))
            inv_xlsx = output_xlsx.parent / f"{output_xlsx.stem}_master_inventory.xlsx"
            st.session_state["inv_xlsx"] = str(inv_xlsx) if inv_xlsx.exists() else None
            st.session_state["inv_csv"] = str(inv_xlsx.with_suffix(".csv")) if inv_xlsx.exists() else None
            st.session_state["run_time"] = elapsed_time

        except Exception as exc:
            st.error(f"❌ حدث خطأ أثناء المعالجة: {exc}")
            logging.exception("Pipeline error in Streamlit")


# ---------------------------------------------------------------------------
# Results & Visual Dashboard
# ---------------------------------------------------------------------------

if "results_groups" in st.session_state:
    groups: list[DuplicateGroup] = st.session_state["results_groups"]
    output_xlsx_path = Path(st.session_state["output_xlsx"])
    output_csv_path = Path(st.session_state["output_csv"])
    inv_xlsx_path = Path(st.session_state["inv_xlsx"]) if st.session_state.get("inv_xlsx") else None

    st.markdown("---")
    st.markdown("<h2 class='rtl-text'>📊 نتائج التحليل ولوحة المؤشرات</h2>", unsafe_allow_html=True)

    # Metrics calculation
    total_images_in_dupes = sum(len(g.members) for g in groups)
    num_groups = len(groups)
    
    total_scanned = total_images_in_dupes
    unique_count = 0
    if inv_xlsx_path and inv_xlsx_path.exists():
        try:
            df_inv = pd.read_excel(inv_xlsx_path)
            total_scanned = len(df_inv)
            unique_count = sum(df_inv["Decision"].astype(str).str.contains("UNIQUE", na=False))
        except Exception:
            pass

    dupe_ratio = (total_images_in_dupes / total_scanned * 100) if total_scanned > 0 else 0

    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{total_scanned:,}</div>
                <div class="metric-label">إجمالي الصور المفحوصة</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with mcol2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{num_groups:,}</div>
                <div class="metric-label">مجموعات التكرار المكتشفة</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with mcol3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value" style="color: #ea580c;">{total_images_in_dupes:,}</div>
                <div class="metric-label">صور داخل مجموعات مكررة</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with mcol4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value" style="color: #16a34a;">{unique_count:,}</div>
                <div class="metric-label">صور فريدة صافية</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with mcol5:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value" style="color: #7c3aed;">{dupe_ratio:.1f}%</div>
                <div class="metric-label">نسبة التكرار في الكتالوج</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Instant Downloads Section
    st.markdown("<h3 class='rtl-text'>📥 تحميل التقارير النهائية للمراجعة البشرية</h3>", unsafe_allow_html=True)
    dcol1, dcol2, dcol3 = st.columns(3)

    with dcol1:
        if output_xlsx_path.exists():
            with open(output_xlsx_path, "rb") as f:
                st.download_button(
                    label="📊 تحميل تقرير المجموعات المتطابقة (Excel)",
                    data=f.read(),
                    file_name="تقرير_مجموعات_التطابق.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

    with dcol2:
        if inv_xlsx_path and inv_xlsx_path.exists():
            with open(inv_xlsx_path, "rb") as f:
                st.download_button(
                    label="📑 تحميل تقرير الكتالوج الشامل (Master Inventory)",
                    data=f.read(),
                    file_name="تقرير_الكتالوج_الشامل_لجميع_الصور.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

    with dcol3:
        if output_csv_path.exists():
            with open(output_csv_path, "rb") as f:
                st.download_button(
                    label="📄 تحميل تقرير CSV",
                    data=f.read(),
                    file_name="تقرير_التطابق.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Duplicate Inspection Gallery
    st.markdown("<h3 class='rtl-text'>👁️ معرض الفحص البصري للمجموعات المتطابقة</h3>", unsafe_allow_html=True)
    st.caption("تصفح المجموعات وتأكد بنفسك من مطابقة المنتجات بدقة جنباً إلى جنب:")

    if num_groups == 0:
        st.info("لم يتم العثور على أي صور مكررة في هذه العينة — جميع الصور فريدة!")
    else:
        fcol1, fcol2 = st.columns([1, 2])
        with fcol1:
            conf_filter = st.selectbox(
                "تصفية المجموعات حسب الثقة:",
                options=["جميع المجموعات", "عالية الثقة جداً (90% فأكثر)", "متوسطة الثقة (70% - 89%)", "تحتاج تدقيق بشري (أقل من 70%)"],
            )

        filtered_groups = groups
        if conf_filter == "عالية الثقة جداً (90% فأكثر)":
            filtered_groups = [g for g in groups if g.confidence >= 90]
        elif conf_filter == "متوسطة الثقة (70% - 89%)":
            filtered_groups = [g for g in groups if 70 <= g.confidence < 90]
        elif conf_filter == "تحتاج تدقيق بشري (أقل من 70%)":
            filtered_groups = [g for g in groups if g.confidence < 70]

        st.write(f"المجموعات المعروضة: **{len(filtered_groups)}** مجموعة")

        page_size = 10
        total_pages = max(1, (len(filtered_groups) + page_size - 1) // page_size)
        current_page = st.number_input("الصفحة:", min_value=1, max_value=total_pages, value=1, step=1)

        start_idx = (current_page - 1) * page_size
        page_groups = filtered_groups[start_idx : start_idx + page_size]

        for grp in page_groups:
            if grp.confidence >= 90:
                badge_class = "badge-confidence-high"
                badge_text = f"ثقة عالية ({grp.confidence}%)"
            elif grp.confidence >= 70:
                badge_class = "badge-confidence-med"
                badge_text = f"ثقة جيدة ({grp.confidence}%)"
            else:
                badge_class = "badge-confidence-low"
                badge_text = f"مراجعة ({grp.confidence}%)"

            match_label = {
                "exact": "تطابق بايتي تام (Exact Copy)",
                "near_exact": "شبه متطابق (مضغوط/مقصوص/علامة مائية)",
                "embedding_color_verified": "نفس الموديل بزاوية/إضاءة مختلفة (مؤكد لوني)",
            }.get(grp.match_type, grp.match_type)

            with st.expander(f"📦 مجموعة رقم #{grp.group_id} — عدد الصور: {len(grp.members)} | {match_label} | {badge_text}", expanded=(len(page_groups) <= 3)):
                img_cols = st.columns(min(len(grp.members), 5))
                for i, member_path in enumerate(grp.members):
                    col = img_cols[i % len(img_cols)]
                    with col:
                        p = Path(member_path)
                        if p.exists():
                            try:
                                img = Image.open(p)
                                st.image(img, use_container_width=True, caption=p.name)
                            except Exception:
                                st.error(f"تعذر فتح: {p.name}")
                        else:
                            st.warning(f"الملف غير متوفر: {p.name}")

    # Interactive Table View
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 class='rtl-text'>📋 جدول تفاصيل الصور والمجموعات</h3>", unsafe_allow_html=True)

    if output_xlsx_path.exists():
        try:
            df_report = pd.read_excel(output_xlsx_path)
            st.dataframe(
                df_report,
                use_container_width=True,
                column_config={
                    "confidence": st.column_config.ProgressColumn(
                        "درجة الثقة",
                        help="0 إلى 100",
                        format="%d%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "file_path": st.column_config.TextColumn("مسار الملف"),
                    "file_name": st.column_config.TextColumn("اسم الملف"),
                    "group_id": st.column_config.NumberColumn("رقم المجموعة"),
                    "match_type": st.column_config.TextColumn("نوع التطابق"),
                    "group_size": st.column_config.NumberColumn("حجم المجموعة"),
                },
            )
        except Exception as exc:
            st.warning(f"تعذر عرض الجدول: {exc}")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; direction: rtl;">
        نظام كشف صور الأثاث المتطابقة والمكررة — مدعوم بنماذج الذكاء الاصطناعي (CLIP ViT, rembg, CIE-Lab, FAISS).
        <br>جميع حقوق الحفظ والخصوصية محفوظة © 2026.
    </div>
    """,
    unsafe_allow_html=True,
)
