"""Bilingual Interactive Web Application for Furniture Product-Image Duplicate Detection.

Designed for local execution, Hugging Face Spaces (16GB RAM Docker), and Streamlit Cloud.
Features a zero-friction 1-Click Built-in Demo, real-time 6-stage progress tracking,
Color Gate False-Positive Guardrail inspection, and audited Kaggle benchmark metrics.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import tempfile
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, load_config  # noqa: E402
from src.pipeline import run as run_pipeline  # noqa: E402
from src.scoring import DuplicateGroup  # noqa: E402

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
DEMO_CATALOG_DIR = PROJECT_ROOT / "demo_catalog"
GITHUB_REPO_URL = "https://github.com/georgewhba/furniture-image-dedup"
KAGGLE_NOTEBOOK_URL = "https://www.kaggle.com/code/djnd27/furniture-duplicate-detection-ikea"

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Furniture Duplicate Detection AI | CLIP + Color Gate",
    page_icon="🪑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load calibrated defaults from config.yaml
BASE_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_CFG = load_config(BASE_CONFIG_PATH) if BASE_CONFIG_PATH.exists() else Config()

if "ui_lang" not in st.session_state:
    st.session_state["ui_lang"] = "en"


def t(en: str, ar: str) -> str:
    """Return localized string based on active UI language."""
    return en if st.session_state.get("ui_lang", "en") == "en" else ar


def is_hosted_cloud() -> bool:
    """Detect whether running inside Hugging Face Spaces, Streamlit Cloud, or Linux container."""
    return (
        bool(os.environ.get("SPACE_ID"))
        or bool(os.environ.get("STREAMLIT_SERVER_BASE_URL"))
        or (not sys.platform.startswith("win"))
    )


# ---------------------------------------------------------------------------
# Language Switcher & Top Links + Dynamic Styling
# ---------------------------------------------------------------------------

top_col_left, top_col_right = st.columns([3, 1.2])
with top_col_right:
    chosen_lang = st.radio(
        "🌐 Language / اللغة",
        options=["English", "العربية"],
        index=0 if st.session_state["ui_lang"] == "en" else 1,
        horizontal=True,
        key="lang_selector",
        label_visibility="collapsed",
    )
    st.session_state["ui_lang"] = "en" if chosen_lang == "English" else "ar"

lang = st.session_state["ui_lang"]
dir_attr = "ltr" if lang == "en" else "rtl"
align_attr = "left" if lang == "en" else "right"

with top_col_left:
    st.markdown(
        f"""
        <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap; direction:{dir_attr}; font-size:0.88rem; color:#475569;">
            <span>🔗 <a href="{GITHUB_REPO_URL}" target="_blank" style="text-decoration:none; color:#1d4ed8; font-weight:600;">GitHub Repository</a></span>
            <span>•</span>
            <span>📊 <a href="{KAGGLE_NOTEBOOK_URL}" target="_blank" style="text-decoration:none; color:#0f766e; font-weight:600;">Kaggle GPU Benchmark Notebook</a></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

cairo_import = (
    "@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');"
    if lang == "ar"
    else "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');"
)
font_stack = (
    "'Cairo', -apple-system, BlinkMacSystemFont, sans-serif"
    if lang == "ar"
    else "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
)

st.markdown(
    f"""
    <style>
    {cairo_import}

    html, body, [class*="css"] {{
        font-family: {font_stack} !important;
        letter-spacing: normal !important;
    }}

    .dir-text {{
        direction: {dir_attr};
        text-align: {align_attr};
    }}

    .hero-banner {{
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 52%, #0d9488 100%);
        color: #ffffff;
        padding: 1.85rem 2.2rem;
        border-radius: 16px;
        box-shadow: 0 12px 28px -8px rgba(15, 23, 42, 0.25);
        margin-bottom: 1.5rem;
        direction: {dir_attr};
        text-align: {align_attr};
    }}
    .hero-title {{
        font-size: 2.05rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
        color: #ffffff;
        line-height: 1.25;
    }}
    .hero-subtitle {{
        font-size: 1.0rem;
        font-weight: 400;
        color: #e0f2fe;
        line-height: 1.6;
        max-width: 980px;
    }}
    .hero-badges {{
        margin-top: 1rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }}
    .hero-badge {{
        background: rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(8px);
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }}

    .metric-card {{
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        text-align: center;
    }}
    .metric-value {{
        font-size: 1.9rem;
        font-weight: 800;
        color: #1e3a8a;
    }}
    .metric-label {{
        font-size: 0.85rem;
        color: #475569;
        font-weight: 600;
        margin-top: 0.2rem;
    }}

    .guardrail-card {{
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 5px solid #e11d48;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        direction: {dir_attr};
        text-align: {align_attr};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-title">{t("🪑 Furniture Duplicate Detection AI", "🪑 نظام كشف وتصفية صور الأثاث المتطابقة بالذكاء الاصطناعي")}</div>
        <div class="hero-subtitle">
            {t(
                "A production 6-stage computer-vision pipeline uniting perceptual hashing, CLIP ViT-B/32 + FAISS semantic retrieval, and a <b>mandatory CIE-Lab Earth Mover's Distance (EMD) & 16×16 spatial Color Gate</b>. It merges exact, cropped, watermarked, and re-shot product photos while strictly refusing to merge identical models in different colors.",
                "نظام رؤية حاسوبية من 6 مراحل يجمع بين البصمة الإدراكية (pHash/dHash)، واسترجاع المعاني عبر CLIP ViT-B/32 وFAISS، و<b>بوابة تحقق لونية إلزامية (CIE-Lab EMD وفحص الشبكة المكانية 16×16)</b>؛ يدمج النسخ المطابقة والمقصوصة والمصورة بزوايا مختلفة بينما يمنع دمج نفس قطعة الأثاث بألوان مختلفة.",
            )}
        </div>
        <div class="hero-badges">
            <span class="hero-badge">{t("🔒 100% Non-Destructive (Read-Only)", "🔒 أمان 100% (لا يحذف الصور)")}</span>
            <span class="hero-badge">{t("🎨 CIE-Lab EMD & Spatial Color Gate", "🎨 بوابة ألوان CIE-Lab EMD")}</span>
            <span class="hero-badge">{t("⚡ CLIP ViT-B/32 + FAISS", "⚡ CLIP ViT-B/32 + FAISS")}</span>
            <span class="hero-badge">{t("🏆 Audited: 95.8% IKEA-FS | 100% RoyalOak", "🏆 نتائج موثقة: 95.8% IKEA-FS | 100% RoyalOak")}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar Settings (Synced to config.yaml)
# ---------------------------------------------------------------------------

st.sidebar.markdown(f"### {t('⚙️ Pipeline Configuration', '⚙️ إعدادات وخيارات الفرز')}")

sample_options = {
    "all": t(
        "All images in dataset (Recommended for Demo / <=500)",
        "كافة الصور (مناسب للعينة أو حتى 500 صورة)",
    ),
    "100": t("Quick sample (100 images)", "عينة سريعة (100 صورة)"),
    "500": t("Medium sample (500 images)", "عينة متوسطة (500 صورة)"),
    "1000": t("Large sample (1,000 images)", "عينة موسعة (1,000 صورة)"),
    "custom": t("Custom limit", "عدد مخصص"),
}

selected_scope = st.sidebar.selectbox(
    t("Processing Scope:", "نطاق المعالجة:"),
    options=list(sample_options.keys()),
    format_func=lambda k: sample_options[k],
    index=0,
)

sample_size: int | None = None
if selected_scope == "100":
    sample_size = 100
elif selected_scope == "500":
    sample_size = 500
elif selected_scope == "1000":
    sample_size = 1000
elif selected_scope == "custom":
    sample_size = int(
        st.sidebar.number_input(
            t("Max images to scan:", "الحد الأقصى للصور:"),
            min_value=5,
            max_value=100000,
            value=200,
            step=25,
        )
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"### {t('🎨 Color Gate & Similarity Thresholds', '🎨 عتبات فحص اللون والتشابه')}"
)
st.sidebar.caption(
    t(
        "Core guardrail: prevents merging identical sofa/chair shapes upholstered in different colors.",
        "الشرط الجوهري: منع تصنيف نفس الكنبة أو الكرسي كتكرار عند اختلاف اللون.",
    )
)

use_bg_removal = st.sidebar.checkbox(
    t("Isolate furniture via rembg (u2netp)", "عزل خلفية المنتج بالذكاء الاصطناعي (rembg)"),
    value=bool(DEFAULT_CFG.use_background_removal),
    help=t(
        "Removes studio/room backgrounds before extracting CIE-Lab color palettes.",
        "يعزل الأثاث عن خلفية الغرفة قبل استخراج الألوان السائدة.",
    ),
)

color_distance_threshold = st.sidebar.slider(
    t("Color Gate Max Delta-E (Lab EMD):", "عتبة المسافة اللونية (Lab EMD ΔE):"),
    min_value=8.0,
    max_value=40.0,
    value=float(DEFAULT_CFG.color_distance_threshold),
    step=0.5,
    help=t(
        "Calibrated default: 16.5. Lower values enforce stricter color matching.",
        "القيمة المعايرة: 16.5. القيم الأقل تزيد صرامة مطابقة اللون.",
    ),
)

similarity_threshold = st.sidebar.slider(
    t("CLIP Shape Similarity Threshold:", "عتبة تشابه الشكل والموديل (CLIP):"),
    min_value=0.75,
    max_value=0.98,
    value=float(DEFAULT_CFG.similarity_threshold),
    step=0.005,
    help=t(
        "Calibrated default: 0.925. Cosine similarity in ViT-B/32 embedding space.",
        "القيمة المعايرة: 0.925. درجة التشابه الدلالي في فضاء ViT-B/32.",
    ),
)

phash_threshold = st.sidebar.slider(
    t("Perceptual Hash Distance (pHash):", "عتبة الهاش الإدراكي (pHash):"),
    min_value=2,
    max_value=16,
    value=int(DEFAULT_CFG.phash_threshold),
    step=1,
    help=t(
        "Calibrated default: 8. Catches crops, JPEG recompression, and watermarks.",
        "القيمة المعايرة: 8. يكتشف القص وضغط الصور والعلامات المائية.",
    ),
)

st.sidebar.markdown("---")
enable_resume = st.sidebar.checkbox(
    t("Enable SQLite Feature Cache (--resume)", "تفعيل كاش SQLite للاستئناف السريع"),
    value=True,
)
st.sidebar.info(
    t(
        "Defaults loaded directly from `config.yaml` (audited on 1,812 furniture catalog images).",
        "القيم الافتراضية محمّلة من `config.yaml` ومعايرة على 1,812 صورة أثاث حقيقية.",
    )
)


# ---------------------------------------------------------------------------
# Helper Functions
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


def parse_gdrive_url(url: str) -> tuple[str | None, bool]:
    raw = url.strip()
    if not raw:
        return None, False
    m_folder = re.search(r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)", raw)
    if m_folder:
        return m_folder.group(1), True
    m_file = re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", raw)
    if m_file:
        return m_file.group(1), False
    m_id = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", raw)
    if m_id:
        return m_id.group(1), False
    if re.match(r"^[a-zA-Z0-9_-]{25,}$", raw):
        return raw, False
    return None, False


# ---------------------------------------------------------------------------
# Main Tabs (1-Click Demo First!)
# ---------------------------------------------------------------------------

(
    tab_demo,
    tab_upload,
    tab_urls,
    tab_kaggle,
    tab_folder,
    tab_benchmarks,
) = st.tabs(
    [
        t("🎯 1-Click Live Demo", "🎯 تجربة فورية بضغطة واحدة"),
        t("📦 Upload ZIP", "📦 رفع ملف ZIP"),
        t("🌐 Drive / Direct URL", "🌐 رابط Drive أو URL مباشر"),
        t("📊 Kaggle Dataset", "📊 داتاست Kaggle"),
        t("📁 Local Folder", "📁 مجلد محلي"),
        t("🏆 Verified Benchmarks", "🏆 نتائج التقييم الموثقة"),
    ]
)

auto_run_demo = False

# TAB 1: Built-in 1-Click Demo Catalog
with tab_demo:
    demo_exists = DEMO_CATALOG_DIR.is_dir() and count_supported_images(DEMO_CATALOG_DIR) > 0
    demo_count = count_supported_images(DEMO_CATALOG_DIR) if demo_exists else 0

    st.markdown(
        f"<div class='dir-text'><b>{t('Zero-Setup Reviewer Sandbox:', 'بيئة تجربة فورية بدون رفع ملفات:')}</b> "
        f"{t(f'Run the complete 6-stage AI pipeline on our curated <b>{demo_count}-image furniture catalog</b> with real e-commerce edge cases built in.', f'شغّل مراحل الذكاء الاصطناعي الست كاملة على كتالوج تجريبي جاهز يضم <b>{demo_count} صورة أثاث</b> مصممة لاختبار أصعب الحالات.')}</div>",
        unsafe_allow_html=True,
    )

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.success(
            t(
                "**✅ Should Merge (True Duplicates)**\n"
                "- Exact byte copy (Bouclé Sofa)\n"
                "- Crop + resize + watermark (Leather Armchair)\n"
                "- Lifestyle room re-shot (Mustard Sofa, 2 angles)\n"
                "- Studio angle variation (Navy Velvet Sofa)",
                "**✅ تكرارات حقيقية (يجب دمجها)**\n"
                "- نسخة مطابقة بالبايت (Bouclé Sofa)\n"
                "- قص + تصغير + علامة مائية (Leather Chair)\n"
                "- مشهد غرفة بزاويتين (Mustard Sofa)\n"
                "- تصوير استوديو بزاوية مختلفة (Navy Sofa)",
            )
        )
    with col_d2:
        st.error(
            t(
                "**🛡️ Must BLOCK (Color Variant Traps)**\n"
                "- **Trap A:** Navy Velvet Sofa vs **Crimson Red** variant\n"
                "- **Trap B:** Living-room Mustard Sofa vs **Teal Blue** variant\n\n"
                "*CLIP shape similarity > 92%, blocked by Color Gate!*",
                "**🛡️ مصائد ألوان (يجب رفض دمجها)**\n"
                "- **مصيدة أ:** الكنبة الكحلي vs نسخة **حمراء** بنفس الشكل\n"
                "- **مصيدة ب:** كنبة الغرفة الصفراء vs نسخة **تركواز** بنفس الغرفة\n\n"
                "*CLIP يتجاوز 92% تشابه، وبوابة الألوان توقف الدمج!*",
            )
        )
    with col_d3:
        st.info(
            t(
                "**🔍 Unique Products (Stay Unmatched)**\n"
                "- Grey Upholstered Armchair\n"
                "- Walnut Wood Coffee Table\n"
                "- Black Oak Storage Coffee Table",
                "**🔍 منتجات فريدة (تبقى منفصلة)**\n"
                "- كرسي قماش رمادي\n"
                "- ترابيزة قهوة خشب جوز\n"
                "- ترابيزة قهوة بلوط أسود",
            )
        )

    if demo_exists:
        demo_btn_col1, demo_btn_col2, demo_btn_col3 = st.columns([1, 2, 1])
        with demo_btn_col2:
            if st.button(
                t(
                    f"🎯 Load & Run 1-Click Live Demo ({demo_count} Furniture Images)",
                    f"🎯 تشغيل التجربة الفورية الآن ({demo_count} صورة أثاث)",
                ),
                type="primary",
                use_container_width=True,
                key="btn_one_click_demo",
            ):
                st.session_state["target_image_dir"] = str(DEMO_CATALOG_DIR)
                st.session_state["image_count"] = demo_count
                st.session_state["dataset_source_label"] = t(
                    "Built-in Curated Demo Catalog", "الكتالوج التجريبي الجاهز"
                )
                auto_run_demo = True

        with st.expander(
            t(
                f"👀 Preview All {demo_count} Built-in Demo Images",
                f"👀 معاينة الـ {demo_count} صورة التجريبية",
            ),
            expanded=False,
        ):
            demo_imgs = sorted(DEMO_CATALOG_DIR.glob("*.jpg"))
            preview_cols = st.columns(5)
            for idx, img_p in enumerate(demo_imgs):
                with preview_cols[idx % 5]:
                    with Image.open(img_p) as im:
                        st.image(im, caption=img_p.name, use_container_width=True)
    else:
        st.warning(t("Demo catalog directory not found.", "مجلد الصور التجريبية غير موجود."))


# TAB 2: Upload ZIP
with tab_upload:
    st.markdown(
        f"<p class='dir-text'><b>{t('Upload a ZIP archive of furniture product images:', 'ارفع ملف مضغوط ZIP يحتوي على الصور:')}</b></p>",
        unsafe_allow_html=True,
    )
    uploaded_zip = st.file_uploader(
        t("Select a .zip file containing images:", "اختر ملف ZIP:"),
        type=["zip"],
        key="uploader_zip",
    )
    if uploaded_zip is not None:
        if st.session_state.get("last_uploaded_name") != uploaded_zip.name:
            temp_dir = Path(tempfile.mkdtemp(prefix="furniture_upload_"))
            with st.spinner(t("Extracting ZIP archive...", "جاري فك ضغط ملف الصور...")):
                unzipped_count = extract_zip(io.BytesIO(uploaded_zip.read()), temp_dir)
            st.session_state["target_image_dir"] = str(temp_dir)
            st.session_state["last_uploaded_name"] = uploaded_zip.name
            st.session_state["image_count"] = unzipped_count
            st.session_state["dataset_source_label"] = uploaded_zip.name
            st.success(
                t(
                    f"✅ Extracted **{unzipped_count:,}** valid images.",
                    f"✅ تم فك الضغط واستخراج **{unzipped_count:,}** صورة بنجاح.",
                )
            )


# TAB 3: Google Drive & Direct URL
with tab_urls:
    u_col1, u_col2 = st.columns(2)
    with u_col1:
        st.markdown(f"#### {t('🌐 Google Drive Link', '🌐 رابط Google Drive')}")
        st.caption(
            t(
                "Ensure sharing is set to **Anyone with the link**.",
                "تأكد أن إذن المشاركة هو **أي شخص لديه الرابط**.",
            )
        )
        gdrive_url_input = st.text_input(
            t("Google Drive File/Folder URL:", "رابط ملف أو مجلد Google Drive:"),
            placeholder="https://drive.google.com/file/d/.../view?usp=sharing",
            key="gdrive_input",
        )
        if st.button(t("📥 Fetch from Google Drive", "📥 تنزيل من Google Drive"), key="btn_gdrive"):
            if not gdrive_url_input.strip():
                st.warning(t("Please enter a Google Drive URL.", "يرجى إدخال رابط أولاً."))
            else:
                file_id, is_folder = parse_gdrive_url(gdrive_url_input)
                if not file_id:
                    st.error(
                        t("Could not parse Drive ID from URL.", "تعذر استخراج المعرّف من الرابط.")
                    )
                else:
                    try:
                        import gdown

                        temp_dir = Path(tempfile.mkdtemp(prefix="gdrive_dl_"))
                        with st.spinner(
                            t("Downloading from Google Drive...", "جاري التنزيل من Google Drive...")
                        ):
                            if is_folder:
                                gdown.download_folder(id=file_id, output=str(temp_dir), quiet=True)
                            else:
                                zip_target = temp_dir / "dataset.zip"
                                downloaded = gdown.download(
                                    id=file_id, output=str(zip_target), quiet=True, fuzzy=True
                                )
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
                            st.session_state["dataset_source_label"] = "Google Drive"
                            st.success(
                                t(
                                    f"✅ Loaded **{n_imgs:,}** images.",
                                    f"✅ تم تحميل **{n_imgs:,}** صورة.",
                                )
                            )
                        else:
                            st.error(
                                t("No supported images found.", "لم يتم العثور على صور مدعومة.")
                            )
                    except Exception as exc:
                        st.error(f"{t('Download failed', 'خطأ في التنزيل')}: {exc}")

    with u_col2:
        st.markdown(f"#### {t('🔗 Direct ZIP URL', '🔗 رابط تنزيل مباشر لملف ZIP')}")
        st.caption(
            t(
                "Supports GitHub Releases, Dropbox (`?dl=1`), or any direct HTTPS `.zip` URL.",
                "يدعم الروابط المباشرة أو Dropbox (`?dl=1`) أو GitHub Releases.",
            )
        )
        direct_url_input = st.text_input(
            t("Direct .zip URL:", "الرابط المباشر لملف ZIP:"),
            placeholder="https://example.com/catalog.zip",
            key="direct_url_input",
        )
        if st.button(t("📥 Fetch Direct ZIP", "📥 تنزيل وفك الضغط"), key="btn_direct_url"):
            if not direct_url_input.strip():
                st.warning(t("Please enter a URL.", "يرجى إدخال الرابط أولاً."))
            else:
                url = direct_url_input.strip()
                if "dropbox.com" in url and "dl=0" in url:
                    url = url.replace("dl=0", "dl=1")
                try:
                    temp_dir = Path(tempfile.mkdtemp(prefix="direct_dl_"))
                    zip_path = temp_dir / "downloaded.zip"
                    with st.spinner(t("Downloading archive...", "جاري تنزيل الملف...")):
                        resp = requests.get(url, stream=True, timeout=90)
                        resp.raise_for_status()
                        with open(zip_path, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                        unzipped_count = extract_zip(zip_path, temp_dir)
                        zip_path.unlink(missing_ok=True)
                    if unzipped_count > 0:
                        st.session_state["target_image_dir"] = str(temp_dir)
                        st.session_state["image_count"] = unzipped_count
                        st.session_state["dataset_source_label"] = "Direct URL"
                        st.success(
                            t(
                                f"✅ Loaded **{unzipped_count:,}** images.",
                                f"✅ تم تحميل **{unzipped_count:,}** صورة.",
                            )
                        )
                    else:
                        st.error(
                            t(
                                "No supported images found inside archive.",
                                "لا توجد صور مدعومة داخل الأرشيف.",
                            )
                        )
                except Exception as exc:
                    st.error(f"{t('Error', 'خطأ')}: {exc}")


# TAB 4: Kaggle Dataset
with tab_kaggle:
    st.markdown(
        f"<p class='dir-text'><b>{t('Import a public Kaggle dataset:', 'تنزيل داتاست من Kaggle مباشرة:')}</b></p>",
        unsafe_allow_html=True,
    )
    kaggle_slug_input = st.text_input(
        t("Kaggle Dataset Slug or URL:", "معرّف أو رابط الداتاست في Kaggle:"),
        placeholder="brikwerk/ikeafs",
        key="kaggle_input",
    )
    with st.expander(
        t("🔑 Kaggle API Credentials (kaggle.json)", "🔑 بيانات مصادقة Kaggle API"), expanded=False
    ):
        kaggle_file = st.file_uploader("kaggle.json", type=["json"], key="kaggle_json_upload")
        if kaggle_file is not None:
            try:
                k_data = json.load(kaggle_file)
                if "username" in k_data and "key" in k_data:
                    os.environ["KAGGLE_USERNAME"] = k_data["username"]
                    os.environ["KAGGLE_KEY"] = k_data["key"]
                    st.success(f"Authenticated as `{k_data['username']}`")
            except Exception as exc:
                st.error(f"Invalid kaggle.json: {exc}")

        k_col1, k_col2 = st.columns(2)
        with k_col1:
            k_user = st.text_input("Kaggle Username", value=os.environ.get("KAGGLE_USERNAME", ""))
        with k_col2:
            k_key = st.text_input(
                "Kaggle Key", type="password", value=os.environ.get("KAGGLE_KEY", "")
            )

    if st.button(t("📥 Download Kaggle Dataset", "📥 تنزيل الداتاست من Kaggle"), key="btn_kaggle"):
        if not kaggle_slug_input.strip():
            st.warning(t("Enter a Kaggle dataset slug first.", "أدخل اسم الداتاست أولاً."))
        else:
            if k_user.strip():
                os.environ["KAGGLE_USERNAME"] = k_user.strip()
            if k_key.strip():
                os.environ["KAGGLE_KEY"] = k_key.strip()
            slug = kaggle_slug_input.strip()
            match = re.search(r"kaggle\.com/(?:datasets/)?([^/]+/[^/?]+)", slug)
            if match:
                slug = match.group(1)
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi

                api = KaggleApi()
                api.authenticate()
                temp_dir = Path(tempfile.mkdtemp(prefix="kaggle_ds_"))
                with st.spinner(f"Downloading `{slug}`..."):
                    api.dataset_download_files(slug, path=str(temp_dir), unzip=True, quiet=True)
                n_imgs = count_supported_images(temp_dir)
                if n_imgs > 0:
                    st.session_state["target_image_dir"] = str(temp_dir)
                    st.session_state["image_count"] = n_imgs
                    st.session_state["dataset_source_label"] = f"Kaggle ({slug})"
                    st.success(
                        t(f"✅ Loaded **{n_imgs:,}** images.", f"✅ تم تنزيل **{n_imgs:,}** صورة.")
                    )
                else:
                    st.error(t("No supported images found.", "لا توجد صور مدعومة."))
            except Exception as exc:
                st.error(f"Kaggle API error: {exc}")


# TAB 5: Local Folder
with tab_folder:
    if is_hosted_cloud():
        st.info(
            t(
                "ℹ️ You are running on a hosted cloud container. Local disk paths (`C:\\`, `D:\\`) on your personal PC are inaccessible from the cloud — please use **🎯 1-Click Live Demo** or **📦 Upload ZIP**.",
                "ℹ️ أنت تتصفح التطبيق الآن على سيرفر سحابي، لذا لا يمكنه الوصول لمسارات القرص على جهازك الشخصي (`D:\\`). استخدم **🎯 التجربة الفورية** أو **📦 رفع ZIP**، أو شغّل التطبيق محلياً عبر `streamlit run streamlit_app.py`.",
            )
        )
    folder_input = st.text_input(
        t("Local folder path:", "مسار مجلد الصور المحلي:"),
        placeholder=str(DEMO_CATALOG_DIR),
        key="local_folder_input",
    )
    if folder_input.strip():
        clean_input = folder_input.strip().strip('"').strip("'")
        if is_hosted_cloud() and (re.match(r"^[a-zA-Z]:", clean_input) or "\\" in clean_input):
            st.error(
                t(
                    f"`{clean_input}` is a local Windows path that cannot be read from this Linux cloud container.",
                    f"المسار `{clean_input}` مسار محلي على جهازك ولا يمكن قراءته من السيرفر السحابي.",
                )
            )
        else:
            resolved_folder = Path(clean_input)
            if resolved_folder.exists() and resolved_folder.is_dir():
                found_count = count_supported_images(resolved_folder)
                st.session_state["target_image_dir"] = str(resolved_folder)
                st.session_state["image_count"] = found_count
                st.session_state["dataset_source_label"] = resolved_folder.name
                st.success(
                    t(
                        f"✅ Found **{found_count:,}** valid images.",
                        f"✅ تم العثور على **{found_count:,}** صورة مدعومة.",
                    )
                )
            else:
                st.error(t("❌ Folder does not exist.", "❌ المسار غير موجود."))


# TAB 6: Verified Benchmarks (IKEA-FS + RoyalOak)
with tab_benchmarks:
    st.markdown(
        f"<h3 class='dir-text'>{t('🏆 Audited Real-World Benchmarks (Tesla T4 GPU on Kaggle)', '🏆 نتائج التقييم الموثقة على داتا سيت حقيقية (Kaggle Tesla T4 GPU)')}</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        t(
            f"All numbers below are reproducible from the public GPU notebook: [Kaggle Notebook v4]({KAGGLE_NOTEBOOK_URL}) | Source repo: [GitHub]({GITHUB_REPO_URL}).",
            f"جميع الأرقام التالية موثقة وقابلة لإعادة الإنتاج عبر النوتبوك العام على Kaggle: [Kaggle Notebook v4]({KAGGLE_NOTEBOOK_URL}) | الكود المصدري: [GitHub]({GITHUB_REPO_URL}).",
        )
    )

    bench_df = pd.DataFrame(
        [
            {
                "Metric / Scenario": "Total Catalog + Evaluation Images",
                "IKEA-FS (Baseline v1)": "1,141 (1,093 real + 48 GT)",
                "IKEA-FS (Upgraded v4)": "1,141 (1,093 real + 48 GT)",
                "RoyalOak Furniture (Upgraded v4)": "671 (623 clean products + 48 GT)",
            },
            {
                "Metric / Scenario": "Overall Ground-Truth Accuracy (48 pairs)",
                "IKEA-FS (Baseline v1)": "85.4% (41 / 48)",
                "IKEA-FS (Upgraded v4)": "95.8% (46 / 48)",
                "RoyalOak Furniture (Upgraded v4)": "100.0% (48 / 48)",
            },
            {
                "Metric / Scenario": "Exact Byte Copies",
                "IKEA-FS (Baseline v1)": "12 / 12 (100%)",
                "IKEA-FS (Upgraded v4)": "12 / 12 (100%)",
                "RoyalOak Furniture (Upgraded v4)": "12 / 12 (100%)",
            },
            {
                "Metric / Scenario": "4% Crop + Resize + JPEG q70",
                "IKEA-FS (Baseline v1)": "5 / 12 (41.7%)",
                "IKEA-FS (Upgraded v4)": "10 / 12 (83.3%)",
                "RoyalOak Furniture (Upgraded v4)": "12 / 12 (100%)",
            },
            {
                "Metric / Scenario": "Watermarked Near-Duplicates",
                "IKEA-FS (Baseline v1)": "12 / 12 (100%)",
                "IKEA-FS (Upgraded v4)": "12 / 12 (100%)",
                "RoyalOak Furniture (Upgraded v4)": "12 / 12 (100%)",
            },
            {
                "Metric / Scenario": "Color-Variant False Positives Blocked",
                "IKEA-FS (Baseline v1)": "12 / 12 GT (merged Morabo/Härlanda in catalog)",
                "IKEA-FS (Upgraded v4)": "12 / 12 GT + 100% color split in catalog",
                "RoyalOak Furniture (Upgraded v4)": "12 / 12 GT + 98.6% category purity",
            },
        ]
    )
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

    st.info(
        t(
            "**Key Engineering Contributions Audited in v4:**\n"
            "1. Fixed `faiss.IndexIVFFlat` defaulting to `METRIC_L2` instead of `METRIC_INNER_PRODUCT` above 1,000 vectors.\n"
            "2. Added studio-border trimming & asymmetric 4% crop pHash/dHash in Stage 2.\n"
            "3. Upgraded Stage 4 Color Gate with **Earth Mover's Distance (Wasserstein-1 optimal transport)** + **crop-safe 16×16 Lab spatial recolor check**.\n"
            "4. Enforced **cannot-link constraints** in Stage 5 quotient-graph clustering to prevent transitive chaining across color variants.\n"
            "5. Data quality audit on `patilaakash619/royaloak`: discovered & filtered out **678 scraped `(425, 210)` website mega-menu banners** prior to benchmark evaluation.",
            "**أبرز التحسينات الهندسية المختبرة في v4:**\n"
            "1. إصلاح خطأ `faiss.IndexIVFFlat` (استخدام `METRIC_INNER_PRODUCT` بدل `L2` الافتراضي).\n"
            "2. إضافة قص الخلفية السادة لحساب أبعاد المنتج + هاش القص 4% (`phash_crop4` / `dhash_crop4`).\n"
            "3. ترقية بوابة الألوان بـ **Earth Mover's Distance (EMD)** وفحص الشبكة المكانية `16×16` داخل مشاهد الغرف.\n"
            "4. تمرير قيود عدم الربط (**Cannot-Link Constraints**) لمرحلة التجميع لمنع ربط الألوان المختلفة عبر وسطاء.\n"
            "5. تدقيق جودة البيانات في `royaloak`: اكتشاف وفلترة **678 صورة بانر قوائم متكررة** قبل القياس النهائي.",
        )
    )


# ---------------------------------------------------------------------------
# Active Dataset Status & Execution Control
# ---------------------------------------------------------------------------

active_dir_str = st.session_state.get("target_image_dir")
active_count = st.session_state.get("image_count", 0)
active_label = st.session_state.get("dataset_source_label", "")

st.markdown("<br>", unsafe_allow_html=True)
if active_dir_str and Path(active_dir_str).exists():
    st.info(
        t(
            f"📂 **Active Dataset Ready:** `{active_label or Path(active_dir_str).name}` — **{active_count:,}** images ready to scan.",
            f"📂 **الكتالوج الجاهز للفحص:** `{active_label or Path(active_dir_str).name}` — يحتوي على **{active_count:,}** صورة.",
        )
    )
else:
    st.warning(
        t(
            "👆 Click **🎯 Load & Run 1-Click Live Demo** in the first tab to test immediately, or upload your own `.zip` catalog.",
            "👆 اضغط **🎯 تشغيل التجربة الفورية الآن** في التبويب الأول للتجربة خلال ثوانٍ، أو ارفع ملف `ZIP` لصورك.",
        )
    )

run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
with run_col2:
    start_clicked = st.button(
        t("🚀 Run 6-Stage AI Deduplication Pipeline", "🚀 بدء الفرز والتحليل الذكي الآن"),
        type="primary" if active_dir_str else "secondary",
        use_container_width=True,
        disabled=(active_dir_str is None),
        key="btn_run_main",
    )

if (start_clicked or auto_run_demo) and active_dir_str is not None:
    target_path = Path(active_dir_str)
    config = replace(
        DEFAULT_CFG,
        use_background_removal=use_bg_removal,
        color_distance_threshold=float(color_distance_threshold),
        similarity_threshold=float(similarity_threshold),
        phash_threshold=int(phash_threshold),
    )

    run_timestamp = int(time.time())
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_xlsx = output_dir / f"furniture_dedup_report_{run_timestamp}.xlsx"
    cache_dir = PROJECT_ROOT / ".dedup_cache"
    telemetry: dict[str, Any] = {}

    with st.container():
        st.markdown(
            f"<h3 class='dir-text'>{t('⏳ Running 6-Stage AI Pipeline...', '⏳ جاري تنفيذ مراحل الفرز الست...')}</h3>",
            unsafe_allow_html=True,
        )
        status_slot = st.empty()
        prog_bar = st.progress(0.02)
        timer_slot = st.empty()
        t_start = time.time()

        try:
            status_slot.info(
                t("Initializing models & scanning files...", "جاري تهيئة النماذج وفحص الملفات...")
            )

            def update_progress(msg: str, val: float) -> None:
                prog_bar.progress(min(1.0, max(0.0, float(val))))
                elapsed = time.time() - t_start
                status_slot.markdown(f"🔹 **{msg}**")
                timer_slot.caption(
                    t(
                        f"⏱️ Elapsed: {elapsed:.1f}s | Overall progress: {int(val * 100)}%",
                        f"⏱️ الوقت المنقضي: {elapsed:.1f} ثانية | الإنجاز: {int(val * 100)}%",
                    )
                )

            groups = run_pipeline(
                input_dir=target_path,
                output_path=output_xlsx,
                config=config,
                cache_dir=cache_dir,
                sample_size=sample_size,
                resume=enable_resume,
                verbose=False,
                progress_callback=update_progress,
                telemetry_out=telemetry,
                progress_lang=lang,
            )

            prog_bar.progress(1.0)
            elapsed_time = time.time() - t_start
            status_slot.success(
                t(
                    f"🎉 Pipeline complete in **{elapsed_time:.1f}s**! Found **{len(groups)}** duplicate groups.",
                    f"🎉 اكتمل الفرز بنجاح خلال **{elapsed_time:.1f} ثانية**! تم اكتشاف **{len(groups)}** مجموعات تكرار.",
                )
            )
            timer_slot.empty()

            inv_xlsx = output_xlsx.parent / f"{output_xlsx.stem}_all_images_inventory.xlsx"
            st.session_state["results_groups"] = groups
            st.session_state["results_telemetry"] = telemetry
            st.session_state["output_xlsx"] = str(output_xlsx)
            st.session_state["output_csv"] = str(output_xlsx.with_suffix(".csv"))
            st.session_state["inv_xlsx"] = str(inv_xlsx) if inv_xlsx.exists() else None
            st.session_state["inv_csv"] = (
                str(inv_xlsx.with_suffix(".csv")) if inv_xlsx.exists() else None
            )
            st.session_state["run_time"] = elapsed_time

        except Exception as exc:
            st.error(f"❌ {t('Pipeline error', 'خطأ أثناء المعالجة')}: {exc}")
            logging.exception("Pipeline error in Streamlit")


# ---------------------------------------------------------------------------
# Interactive Results Dashboard + Color Gate Guardrail Showcase
# ---------------------------------------------------------------------------

if "results_groups" in st.session_state:
    groups: list[DuplicateGroup] = st.session_state["results_groups"]
    telemetry_res: dict[str, Any] = st.session_state.get("results_telemetry", {})
    output_xlsx_path = Path(st.session_state["output_xlsx"])
    output_csv_path = Path(st.session_state["output_csv"])
    inv_xlsx_path = Path(st.session_state["inv_xlsx"]) if st.session_state.get("inv_xlsx") else None

    st.markdown("---")
    st.markdown(
        f"<h2 class='dir-text'>{t('📊 Deduplication Results & Audit Dashboard', '📊 نتائج الفرز ولوحة المؤشرات')}</h2>",
        unsafe_allow_html=True,
    )

    total_images_in_dupes = sum(len(g.members) for g in groups)
    num_groups = len(groups)
    total_scanned = telemetry_res.get("total_images", total_images_in_dupes)
    unique_count = max(0, total_scanned - total_images_in_dupes)

    if inv_xlsx_path and inv_xlsx_path.exists():
        try:
            df_inv = pd.read_excel(inv_xlsx_path)
            total_scanned = len(df_inv)
            unique_count = int(
                df_inv["Decision"].astype(str).str.contains("UNIQUE", na=False).sum()
            )
        except Exception:
            pass

    dupe_ratio = (total_images_in_dupes / total_scanned * 100) if total_scanned > 0 else 0.0
    blocked_list: list[dict[str, Any]] = telemetry_res.get("rejected_pairs", [])

    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{total_scanned:,}</div>
                <div class="metric-label">{t("Images Scanned", "الصور المفحوصة")}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with mcol2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{num_groups:,}</div>
                <div class="metric-label">{t("Duplicate Groups", "مجموعات التكرار")}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with mcol3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value" style="color: #ea580c;">{total_images_in_dupes:,}</div>
                <div class="metric-label">{t("Grouped Images", "صور داخل مجموعات")}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with mcol4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value" style="color: #16a34a;">{unique_count:,}</div>
                <div class="metric-label">{t("Unique Products", "صور فريدة صافية")}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with mcol5:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value" style="color: #e11d48;">{len(blocked_list):,}</div>
                <div class="metric-label">{t("Color Blocks (FP Saved)", "أخطاء ألوان تم صدّها")}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Report Downloads
    st.markdown(
        f"<h3 class='dir-text'>{t('📥 Download Audit Reports (Non-Destructive Review)', '📥 تحميل التقارير للمراجعة البشرية')}</h3>",
        unsafe_allow_html=True,
    )
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        if output_xlsx_path.exists():
            st.download_button(
                label=t("📊 Duplicate Groups (.xlsx)", "📊 تقرير المجموعات المتطابقة (.xlsx)"),
                data=output_xlsx_path.read_bytes(),
                file_name="furniture_duplicates_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    with dcol2:
        if inv_xlsx_path and inv_xlsx_path.exists():
            st.download_button(
                label=t("📑 Full Master Inventory (.xlsx)", "📑 الجرد الشامل لكل الصور (.xlsx)"),
                data=inv_xlsx_path.read_bytes(),
                file_name="furniture_all_images_inventory.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    with dcol3:
        if output_csv_path.exists():
            st.download_button(
                label=t("📄 Duplicate Groups (.csv)", "📄 تقرير المجموعات (.csv)"),
                data=output_csv_path.read_bytes(),
                file_name="furniture_duplicates_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # Shield Showcase: Color Gate Blocked False Positives
    if blocked_list:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<h3 class='dir-text'>{t('🛡️ Mandatory Color Gate Guardrail — Prevented False Positives', '🛡️ بوابة التحقق اللوني — أزواج متشابهة شكلياً ومرفوضة لاختلاف اللون')}</h3>",
            unsafe_allow_html=True,
        )
        st.caption(
            t(
                "These product pairs scored high shape/structural similarity (pHash or CLIP > threshold) yet were BLOCKED from merging because their CIE-Lab EMD / spatial color distance exceeded the Color Gate limit.",
                "هذه المنتجات حققت تشابهاً عالياً في الشكل والهيكل، لكن بوابة الألوان الإلزامية رفضت دمجها لتجاوزها عتبة المسافة اللونية في فضاء CIE-Lab.",
            )
        )
        for idx, rej in enumerate(blocked_list[:6], start=1):
            pa, pb = Path(rej["path_a"]), Path(rej["path_b"])
            sim_val = rej.get("similarity")
            sim_str = f"{sim_val * 100:.1f}%" if sim_val is not None else "pHash"
            dist_val = rej.get("color_distance", 0.0)
            thresh_val = rej.get("threshold", color_distance_threshold)
            st.markdown(
                f"""<div class="guardrail-card">
                    <b>#{idx} — 🛑 {t("BLOCKED BY COLOR GATE", "تم الرفض عبر بوابة الألوان")}</b> ({rej.get("stage", "")})<br>
                    <code>{pa.name}</code> ↔ <code>{pb.name}</code><br>
                    {t("Shape Similarity:", "تشابه الشكل:")} <b>{sim_str}</b> |
                    {t("Color Delta-E Distance:", "المسافة اللونية ΔE:")} <b style="color:#be123c;">{dist_val:.2f}</b>
                    (&gt; {thresh_val:.1f} {t("threshold", "العتبة")})
                </div>""",
                unsafe_allow_html=True,
            )
            g_c1, g_c2 = st.columns(2)
            with g_c1:
                if pa.exists():
                    with Image.open(pa) as ima:
                        st.image(ima, caption=pa.name, use_container_width=True)
            with g_c2:
                if pb.exists():
                    with Image.open(pb) as imb:
                        st.image(imb, caption=pb.name, use_container_width=True)

    # Visual Duplicate Inspection Gallery
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<h3 class='dir-text'>{t('👁️ Visual Duplicate Groups Inspector', '👁️ معرض الفحص البصري للمجموعات المتطابقة')}</h3>",
        unsafe_allow_html=True,
    )

    if num_groups == 0:
        st.info(
            t(
                "No duplicate groups detected — all images are unique.",
                "لم يتم العثور على أي صور مكررة — جميع الصور فريدة.",
            )
        )
    else:
        fcol1, fcol2 = st.columns(2)
        filter_opts = {
            "all": t("All groups", "جميع المجموعات"),
            "high": t("High confidence (>= 90%)", "ثقة عالية (90% فأكثر)"),
            "med": t("Medium confidence (70% - 89%)", "ثقة متوسطة (70% - 89%)"),
            "low": t("Needs human review (< 70%)", "مراجعة بشرية (أقل من 70%)"),
        }
        with fcol1:
            conf_key = st.selectbox(
                t("Filter by confidence:", "تصفية حسب الثقة:"),
                options=list(filter_opts.keys()),
                format_func=lambda k: filter_opts[k],
            )
        tier_opts = {
            "all": t("All detection stages", "كل مراحل الاكتشاف"),
            "exact": "exact (Stage 1 SHA-256)",
            "near_exact": "near_exact (Stage 2 pHash / Crop)",
            "embedding_color_verified": "embedding_color_verified (Stage 3+4 CLIP + Color Gate)",
        }
        with fcol2:
            tier_key = st.selectbox(
                t("Filter by stage:", "تصفية حسب المرحلة:"),
                options=list(tier_opts.keys()),
                format_func=lambda k: tier_opts[k],
            )

        filtered_groups = groups
        if conf_key == "high":
            filtered_groups = [g for g in filtered_groups if g.confidence >= 90]
        elif conf_key == "med":
            filtered_groups = [g for g in filtered_groups if 70 <= g.confidence < 90]
        elif conf_key == "low":
            filtered_groups = [g for g in filtered_groups if g.confidence < 70]
        if tier_key != "all":
            filtered_groups = [g for g in filtered_groups if g.match_type == tier_key]

        st.write(
            t(
                f"Showing **{len(filtered_groups)}** groups ({dupe_ratio:.1f}% catalog overlap)",
                f"المعروض: **{len(filtered_groups)}** مجموعة (نسبة التكرار {dupe_ratio:.1f}%)",
            )
        )

        page_size = 10
        total_pages = max(1, (len(filtered_groups) + page_size - 1) // page_size)
        current_page = int(
            st.number_input(
                t("Page:", "الصفحة:"), min_value=1, max_value=total_pages, value=1, step=1
            )
        )
        start_idx = (current_page - 1) * page_size
        page_groups = filtered_groups[start_idx : start_idx + page_size]

        match_labels = {
            "exact": t("Exact Byte Copy (SHA-256)", "تطابق بايتي تام (SHA-256)"),
            "near_exact": t(
                "Near-Duplicate (Crop / Recompress / Watermark)", "شبه متطابق (قص/ضغط/علامة مائية)"
            ),
            "embedding_color_verified": t(
                "Same Model, Color-Verified (CLIP + CIE-Lab EMD)",
                "نفس الموديل مؤكد لونياً (CLIP + Lab EMD)",
            ),
        }

        for grp in page_groups:
            tier_desc = match_labels.get(grp.match_type, grp.match_type)
            icon = "🟢" if grp.confidence >= 90 else ("🟡" if grp.confidence >= 70 else "🟠")
            with st.expander(
                f"{icon} Group #{grp.group_id} — {len(grp.members)} {t('images', 'صور')} | {tier_desc} | {t('Confidence', 'الثقة')}: {grp.confidence}%",
                expanded=(len(page_groups) <= 5),
            ):
                img_cols = st.columns(min(len(grp.members), 5))
                for i, member_path in enumerate(grp.members):
                    with img_cols[i % len(img_cols)]:
                        p = Path(member_path)
                        if p.exists():
                            try:
                                with Image.open(p) as img:
                                    st.image(
                                        img,
                                        use_container_width=True,
                                        caption=f"{p.name} ({img.width}×{img.height})",
                                    )
                            except Exception:
                                st.error(f"Cannot open {p.name}")
                        else:
                            st.warning(f"Missing: {p.name}")

    # Tabular Audit Output
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<h3 class='dir-text'>{t('📋 Detailed Group Table', '📋 جدول تفاصيل المجموعات')}</h3>",
        unsafe_allow_html=True,
    )
    if output_xlsx_path.exists():
        try:
            df_report = pd.read_excel(output_xlsx_path)
            st.dataframe(
                df_report,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "group_id": st.column_config.NumberColumn(t("Group ID", "رقم المجموعة")),
                    "file_name": st.column_config.TextColumn(t("File Name", "اسم الملف")),
                    "confidence": st.column_config.ProgressColumn(
                        t("Confidence", "درجة الثقة"),
                        format="%d%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "match_type": st.column_config.TextColumn(t("Match Stage", "نوع التطابق")),
                    "group_size": st.column_config.NumberColumn(t("Group Size", "حجم المجموعة")),
                    "file_path": st.column_config.TextColumn(t("File Path", "مسار الملف")),
                },
            )
        except Exception as exc:
            st.warning(f"Could not render table: {exc}")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; direction:{dir_attr};">
        {t(
            f"Furniture Product-Image Duplicate Detection AI • OpenCLIP ViT-B/32, FAISS, rembg (u2netp) & CIE-Lab EMD Color Gate • <a href='{GITHUB_REPO_URL}' target='_blank'>GitHub</a> • <a href='{KAGGLE_NOTEBOOK_URL}' target='_blank'>Kaggle Notebook</a>",
            f"نظام كشف صور الأثاث المتطابقة — مدعوم بـ OpenCLIP ViT-B/32 وFAISS وrembg وبوابة ألوان CIE-Lab EMD • <a href='{GITHUB_REPO_URL}' target='_blank'>GitHub</a> • <a href='{KAGGLE_NOTEBOOK_URL}' target='_blank'>Kaggle Notebook</a>",
        )}
    </div>
    """,
    unsafe_allow_html=True,
)
