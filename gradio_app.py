"""Interactive Gradio Web Application for Furniture Product-Image Duplicate Detection.

Built natively for Gradio 6.x + Hugging Face Spaces (ZeroGPU & CPU) and local execution.
Runs the 6-stage deduplication pipeline (SHA-256 -> pHash/Crop4 -> OpenCLIP ViT-B/32 + FAISS ->
rembg + CIE-Lab EMD + Spatial Color Gate -> Quotient Grouping -> Excel/CSV Reports) while
keeping source catalogs untouched and exposing Color Gate false-positive rejections.
"""

from __future__ import annotations

import html
import logging
import shutil
import tempfile
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

try:
    import spaces  # type: ignore[import-not-found]
except ImportError:
    spaces = None

from src.config import Config, load_config
from src.pipeline import run as run_pipeline
from src.scoring import DuplicateGroup

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEMO_CATALOG_DIR = PROJECT_ROOT / "demo_catalog"

GITHUB_REPO_URL = "https://github.com/georgewhba/furniture-image-dedup"
KAGGLE_NOTEBOOK_URL = "https://www.kaggle.com/code/djnd27/furniture-duplicate-detection-ikea"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

CUSTOM_CSS = """
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #0e7490 100%);
    color: #f8fafc;
    padding: 24px 28px;
    border-radius: 14px;
    margin-bottom: 14px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.14);
}
.hero-banner h1 {
    margin: 0 0 8px 0;
    font-size: 1.75rem;
    font-weight: 800;
    color: #ffffff;
}
.hero-banner p {
    margin: 0 0 10px 0;
    color: #e2e8f0;
    font-size: 0.98rem;
    line-height: 1.5;
}
.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}
.hero-badge {
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.28);
    color: #ffffff !important;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    text-decoration: none;
}
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin: 12px 0;
}
.kpi-card {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-top: 4px solid #2563eb !important;
    border-radius: 10px;
    padding: 12px 14px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
    color: #0f172a !important;
}
.kpi-card.shield {
    border-top-color: #e11d48 !important;
    background: #fff1f2 !important;
}
.kpi-card.success {
    border-top-color: #059669 !important;
    background: #ecfdf5 !important;
}
.kpi-val {
    font-size: 1.65rem;
    font-weight: 800;
    color: #0f172a !important;
    line-height: 1.2;
}
.kpi-card.success .kpi-val {
    color: #047857 !important;
}
.kpi-card.shield .kpi-val {
    color: #be123c !important;
}
.kpi-label {
    font-size: 0.82rem;
    color: #334155 !important;
    margin-top: 4px;
    font-weight: 600;
}
.guardrail-card {
    background: #fff1f2 !important;
    border: 1px solid #fda4af !important;
    border-left: 5px solid #e11d48 !important;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    color: #0f172a !important;
    line-height: 1.55;
}
.guardrail-card b,
.guardrail-card span,
.guardrail-card div {
    color: #0f172a !important;
}
.guardrail-card b.delta-e-val {
    color: #be123c !important;
}
.guardrail-card code {
    background: #ffe4e6 !important;
    color: #881337 !important;
    padding: 2px 6px !important;
    border-radius: 5px !important;
    font-weight: 600 !important;
    border: 1px solid #fecdd3 !important;
}
.stage-funnel {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.88rem;
    color: #1e293b !important;
    margin-top: 8px;
}
.stage-funnel b,
.stage-funnel span {
    color: #0f172a !important;
}
.rtl-box {
    direction: rtl;
    text-align: right;
}
"""


def _tr(lang: str, en: str, ar: str) -> str:
    """Return bilingual UI copy for ``en`` or ``ar``."""
    return ar if lang == "ar" else en


def _gpu_decorator(fn: Any) -> Any:
    """Attach Hugging Face ZeroGPU decorator when running inside a ZeroGPU runtime."""
    if spaces is not None and hasattr(spaces, "GPU"):
        return spaces.GPU(duration=25)(fn)
    return fn


def _load_default_config() -> Config:
    """Load baseline configuration cleanly with fallback defaults."""
    if DEFAULT_CONFIG_PATH.is_file():
        return load_config(DEFAULT_CONFIG_PATH)
    return Config()


def _list_demo_images() -> list[tuple[str, str]]:
    """Return preview gallery pairs for the curated demo catalog."""
    if not DEMO_CATALOG_DIR.is_dir():
        return []
    return [(str(p), p.name) for p in sorted(DEMO_CATALOG_DIR.glob("*.jpg"))]


def _warm_caches_safely() -> None:
    """Warm CLIP / rembg weights on disk at CPU startup so first request is fast."""
    try:
        import huggingface_hub

        huggingface_hub.hf_hub_download(
            repo_id="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            filename="open_clip_model.safetensors",
        )
    except Exception as exc:
        logger.debug("Skipped OpenCLIP cache warmup: %s", exc)
    try:
        from rembg import new_session

        new_session("u2netp")
    except Exception as exc:
        logger.debug("Skipped rembg u2netp warmup: %s", exc)


def _stage_uploaded_files(uploaded_files: list[str] | None) -> tuple[Path, int]:
    """Copy uploaded images or safely unpack ZIP archives into an isolated temp directory."""
    staging_dir = Path(tempfile.mkdtemp(prefix="hf_dedup_input_"))
    if not uploaded_files:
        return staging_dir, 0

    copied = 0
    for item in uploaded_files:
        src = Path(item)
        if not src.is_file():
            continue
        if src.suffix.lower() == ".zip":
            dest_root = staging_dir.resolve()
            with zipfile.ZipFile(src, "r") as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    rel = Path(member.filename)
                    if rel.is_absolute() or ".." in rel.parts:
                        continue
                    target = (dest_root / rel).resolve()
                    if not str(target).startswith(str(dest_root)):
                        continue
                    if target.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src_f, open(target, "wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f)
                    copied += 1
        elif src.suffix.lower() in IMAGE_EXTENSIONS:
            target = staging_dir / src.name
            if target.exists():
                target = staging_dir / f"{copied}_{src.name}"
            shutil.copy2(src, target)
            copied += 1
    return staging_dir, copied


def _format_summary_html(
    lang: str,
    telemetry: dict[str, Any],
    groups: list[DuplicateGroup],
    runtime_sec: float,
    source_label: str,
) -> str:
    """Build bilingual HTML executive KPI cards and stage telemetry bar."""
    total_images = int(telemetry.get("total_images", 0))
    images_in_groups = sum(len(g.members) for g in groups)
    unique_images = max(0, total_images - images_in_groups)
    blocked = len(telemetry.get("rejected_pairs", []))
    dir_cls = " rtl-box" if lang == "ar" else ""

    kpis = f"""
    <div class="kpi-grid{dir_cls}">
        <div class="kpi-card">
            <div class="kpi-val">{total_images:,}</div>
            <div class="kpi-label">{_tr(lang, "Images Scanned", "إجمالي الصور المفحوصة")}</div>
        </div>
        <div class="kpi-card success">
            <div class="kpi-val">{len(groups):,}</div>
            <div class="kpi-label">{_tr(lang, "Duplicate Groups Merged", "مجموعات التكرار المدمجة")}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val">{unique_images:,}</div>
            <div class="kpi-label">{_tr(lang, "Unique / Unmatched", "صور فريدة منفصلة")}</div>
        </div>
        <div class="kpi-card shield">
            <div class="kpi-val">{blocked:,}</div>
            <div class="kpi-label">{_tr(lang, "Color Traps Blocked", "مصائد الألوان المرفوضة")}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val">{runtime_sec:.1f}s</div>
            <div class="kpi-label">{_tr(lang, "Pipeline Runtime", "زمن التشغيل")}</div>
        </div>
    </div>
    <div class="stage-funnel{dir_cls}">
        <b>{_tr(lang, "Source:", "المصدر:")}</b> {html.escape(source_label)} &nbsp;|&nbsp;
        <b>Stage 1 (SHA-256):</b> {telemetry.get("stage1_exact_groups", 0)} {_tr(lang, "groups", "مجموعات")} &nbsp;•&nbsp;
        <b>Stage 2 (pHash):</b> {telemetry.get("stage2_phash_candidates", 0)} {_tr(lang, "candidates", "مرشح")} &nbsp;•&nbsp;
        <b>Stage 3 (CLIP+FAISS):</b> {telemetry.get("stage3_clip_candidates", 0)} {_tr(lang, "candidates", "مرشح")} &nbsp;•&nbsp;
        <b>Stage 4 (Color Gate):</b> {telemetry.get("stage4_passed", 0)} {_tr(lang, "passed", "ناجح")} / {telemetry.get("stage4_rejected", 0)} {_tr(lang, "blocked", "مرفوض")}
    </div>
    """
    return kpis


def _format_guardrail_outputs(
    lang: str,
    rejected_pairs: list[dict[str, Any]],
    threshold: float,
) -> tuple[str, list[tuple[str, str]]]:
    """Render HTML explanation cards and side-by-side gallery tuples for blocked pairs."""
    if not rejected_pairs:
        msg = _tr(
            lang,
            "No color-variant traps triggered the Shape-vs-Color Guardrail in this run.",
            "لم تظهر أزواج متشابهة شكلياً ومختلفة لونياً في هذه التشغيلية.",
        )
        return f"<div class='stage-funnel'>{msg}</div>", []

    cards: list[str] = []
    gallery_items: list[tuple[str, str]] = []
    for idx, rej in enumerate(rejected_pairs[:8], start=1):
        pa = Path(rej["path_a"])
        pb = Path(rej["path_b"])
        sim_val = rej.get("similarity")
        sim_str = f"{sim_val * 100:.1f}%" if sim_val is not None else "pHash Match"
        dist_val = float(rej.get("color_distance", 0.0))
        t_val = float(rej.get("threshold", threshold))
        stage_name = html.escape(str(rej.get("stage", "Color Gate")))
        dir_cls = " rtl-box" if lang == "ar" else ""
        cards.append(f"""<div class="guardrail-card{dir_cls}">
                <b>#{idx} — 🛑 {_tr(lang, "BLOCKED BY COLOR GATE", "تم الرفض عبر بوابة الألوان")}</b> ({stage_name})<br>
                <code>{html.escape(pa.name)}</code> ↔ <code>{html.escape(pb.name)}</code><br>
                {_tr(lang, "Shape Similarity:", "تشابه الشكل:")} <b>{sim_str}</b> &nbsp;|&nbsp;
                {_tr(lang, "CIE-Lab Color Distance ΔE:", "المسافة اللونية ΔE:")} <b class="delta-e-val">{dist_val:.2f}</b>
                (&gt; {t_val:.1f} {_tr(lang, "threshold", "العتبة")})
            </div>""")
        if pa.is_file():
            gallery_items.append((str(pa), f"Trap #{idx}A ({sim_str}): {pa.name}"))
        if pb.is_file():
            gallery_items.append(
                (
                    str(pb),
                    f"Trap #{idx}B — BLOCKED (ΔE={dist_val:.1f} > {t_val:.1f}): {pb.name}",
                )
            )
    return "\n".join(cards), gallery_items


def _execute_pipeline_core(
    input_dir_str: str,
    output_xlsx_str: str,
    cache_dir_str: str,
    use_bg_removal: bool,
    color_threshold: float,
    sim_threshold: float,
    phash_threshold: int,
    lang: str,
    source_label: str,
    embedding_device: str = "cpu",
) -> tuple[
    str,
    str,
    list[tuple[str, str]],
    list[tuple[str, str]],
    pd.DataFrame,
    str | None,
    str | None,
    str | None,
]:
    """Run the 6-stage deduplication pipeline inside ZeroGPU or local/Space CPU."""
    base_cfg = _load_default_config()
    cfg = replace(
        base_cfg,
        embedding_device=embedding_device,
        use_background_removal=bool(use_bg_removal),
        color_distance_threshold=float(color_threshold),
        similarity_threshold=float(sim_threshold),
        phash_threshold=int(phash_threshold),
    )

    output_xlsx = Path(output_xlsx_str)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    telemetry: dict[str, Any] = {}

    t0 = time.time()
    groups = run_pipeline(
        input_dir=Path(input_dir_str),
        output_path=output_xlsx,
        cache_dir=Path(cache_dir_str),
        config=cfg,
        resume=False,
        verbose=False,
        telemetry_out=telemetry,
        progress_lang=lang,
    )
    runtime_sec = time.time() - t0

    summary_html = _format_summary_html(lang, telemetry, groups, runtime_sec, source_label)
    guardrail_html, guardrail_gallery = _format_guardrail_outputs(
        lang,
        telemetry.get("rejected_pairs", []),
        float(color_threshold),
    )

    tier_labels = {
        "exact": _tr(lang, "Exact (SHA-256)", "تطابق تام"),
        "near_exact": _tr(lang, "Near-Exact (pHash/Crop)", "شبه متطابق"),
        "embedding_color_verified": _tr(
            lang, "Cross-Angle Verified (CLIP+Color)", "تطابق زاوية/إعادة تصوير"
        ),
    }
    group_gallery: list[tuple[str, str]] = []
    for grp in groups:
        tier_text = tier_labels.get(grp.match_type, grp.match_type)
        for m in grp.members:
            p = Path(m)
            if p.is_file():
                group_gallery.append(
                    (
                        str(p),
                        f"Group #{grp.group_id} | {tier_text} ({grp.confidence}%) | {p.name}",
                    )
                )

    csv_path = output_xlsx.with_suffix(".csv")
    inv_xlsx = output_xlsx.parent / f"{output_xlsx.stem}_all_images_inventory.xlsx"

    if csv_path.is_file():
        df_table = pd.read_csv(csv_path)
    else:
        df_table = pd.DataFrame(
            columns=[
                "group_id",
                "file_name",
                "confidence",
                "match_type",
                "group_size",
                "file_path",
            ]
        )

    return (
        summary_html,
        guardrail_html,
        guardrail_gallery,
        group_gallery,
        df_table,
        str(output_xlsx) if output_xlsx.is_file() else None,
        str(csv_path) if csv_path.is_file() else None,
        str(inv_xlsx) if inv_xlsx.is_file() else None,
    )


def _run_builtin_demo_impl(
    use_bg_removal: bool,
    color_threshold: float,
    sim_threshold: float,
    phash_threshold: int,
    lang: str,
    embedding_device: str,
) -> tuple[Any, ...]:
    """Execute the 6-stage pipeline directly on the bundled 13-image demo catalog."""
    if not DEMO_CATALOG_DIR.is_dir():
        raise gr.Error(
            _tr(lang, "demo_catalog folder not found.", "مجلد الصور التجريبية غير موجود.")
        )

    run_root = Path(tempfile.mkdtemp(prefix="hf_demo_out_"))
    output_xlsx = run_root / "furniture_demo_report.xlsx"
    cache_dir = run_root / ".dedup_cache"
    label = _tr(
        lang, "Built-in Curated Demo Catalog (13 Images)", "الكتالوج التجريبي الجاهز (13 صورة)"
    )

    return _execute_pipeline_core(
        str(DEMO_CATALOG_DIR),
        str(output_xlsx),
        str(cache_dir),
        use_bg_removal,
        color_threshold,
        sim_threshold,
        phash_threshold,
        lang,
        label,
        embedding_device=embedding_device,
    )


def run_builtin_demo(
    use_bg_removal: bool,
    color_threshold: float,
    sim_threshold: float,
    phash_threshold: int,
    lang: str,
) -> tuple[Any, ...]:
    """Run demo catalog on CPU (unlimited free runs, zero ZeroGPU quota consumed)."""
    return _run_builtin_demo_impl(
        use_bg_removal,
        color_threshold,
        sim_threshold,
        phash_threshold,
        lang,
        embedding_device="cpu",
    )


@_gpu_decorator
def run_builtin_demo_gpu(
    use_bg_removal: bool,
    color_threshold: float,
    sim_threshold: float,
    phash_threshold: int,
    lang: str,
) -> tuple[Any, ...]:
    """Optional ZeroGPU-accelerated demo runner (short 25s A10G lease)."""
    return _run_builtin_demo_impl(
        use_bg_removal,
        color_threshold,
        sim_threshold,
        phash_threshold,
        lang,
        embedding_device="auto",
    )


def _run_uploaded_catalog_impl(
    uploaded_files: list[str] | None,
    use_bg_removal: bool,
    color_threshold: float,
    sim_threshold: float,
    phash_threshold: int,
    lang: str,
    embedding_device: str,
) -> tuple[Any, ...]:
    """Stage uploaded ZIP/images cleanly and execute the 6-stage pipeline."""
    staging_dir, count = _stage_uploaded_files(uploaded_files)
    if count < 2:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise gr.Error(
            _tr(
                lang,
                "Please upload at least 2 images (or a .zip archive containing 2+ images).",
                "يرجى رفع صورتين على الأقل (أو ملف ZIP يحتوي على صورتين فأكثر).",
            )
        )

    run_root = Path(tempfile.mkdtemp(prefix="hf_upload_out_"))
    output_xlsx = run_root / "furniture_dedup_report.xlsx"
    cache_dir = run_root / ".dedup_cache"
    label = _tr(lang, f"Uploaded Catalog ({count} Images)", f"كتالوج مرفوع ({count} صورة)")

    return _execute_pipeline_core(
        str(staging_dir),
        str(output_xlsx),
        str(cache_dir),
        use_bg_removal,
        color_threshold,
        sim_threshold,
        phash_threshold,
        lang,
        label,
        embedding_device=embedding_device,
    )


def run_uploaded_catalog(
    uploaded_files: list[str] | None,
    use_bg_removal: bool,
    color_threshold: float,
    sim_threshold: float,
    phash_threshold: int,
    lang: str,
) -> tuple[Any, ...]:
    """Deduplicate uploaded images on CPU (unlimited free runs, no ZeroGPU quota block)."""
    return _run_uploaded_catalog_impl(
        uploaded_files,
        use_bg_removal,
        color_threshold,
        sim_threshold,
        phash_threshold,
        lang,
        embedding_device="cpu",
    )


@_gpu_decorator
def run_uploaded_catalog_gpu(
    uploaded_files: list[str] | None,
    use_bg_removal: bool,
    color_threshold: float,
    sim_threshold: float,
    phash_threshold: int,
    lang: str,
) -> tuple[Any, ...]:
    """Optional ZeroGPU-accelerated catalog upload runner (short 25s A10G lease)."""
    return _run_uploaded_catalog_impl(
        uploaded_files,
        use_bg_removal,
        color_threshold,
        sim_threshold,
        phash_threshold,
        lang,
        embedding_device="auto",
    )


def _hero_html(lang: str) -> str:
    """Build bilingual hero banner HTML."""
    dir_cls = " rtl-box" if lang == "ar" else ""
    if lang == "ar":
        return f"""
        <div class="hero-banner{dir_cls}">
            <h1>🪑 نظام كشف صور الأثاث المتطابقة (مدعوم ببوابة التحقق اللوني)</h1>
            <p>خط معالجة مكوّن من 6 مراحل يجمع بين SHA-256 و pHash/dHash و OpenCLIP ViT-B/32 و FAISS، مع بوابة تحقق لوني إلزامية (rembg u2netp + CIE-Lab EMD) لمنع دمج المنتجات المتطابقة في الشكل والمختلفة في اللون. <b>لا يمسح ولا يعدّل أي صورة أصلية.</b></p>
            <div class="hero-badges">
                <span class="hero-badge">⚡ جاهز لتجربة بضغطة زر</span>
                <span class="hero-badge">🛡️ بوابة ألوان CIE-Lab EMD</span>
                <a class="hero-badge" href="{GITHUB_REPO_URL}" target="_blank">📦 مستودع GitHub</a>
                <a class="hero-badge" href="{KAGGLE_NOTEBOOK_URL}" target="_blank">📊 تجارب Kaggle GPU</a>
            </div>
        </div>
        """
    return f"""
    <div class="hero-banner">
        <h1>🪑 Furniture Product-Image Duplicate Detection AI</h1>
        <p>A 6-stage e-commerce catalog deduplication pipeline combining SHA-256, Perceptual Hash, OpenCLIP ViT-B/32 + FAISS, and a <b>mandatory CIE-Lab Earth Mover&rsquo;s Distance Color Gate</b> to prevent false-positive merges on same-shape, different-color SKUs. <b>Zero destructive edits.</b></p>
        <div class="hero-badges">
            <span class="hero-badge">⚡ ZeroGPU / CPU Ready</span>
            <span class="hero-badge">🛡️ Shape-vs-Color Guardrail</span>
            <a class="hero-badge" href="{GITHUB_REPO_URL}" target="_blank">📦 GitHub Repository</a>
            <a class="hero-badge" href="{KAGGLE_NOTEBOOK_URL}" target="_blank">📊 Kaggle GPU Benchmarks</a>
        </div>
    </div>
    """


def _demo_info_md(lang: str) -> str:
    """Build bilingual Markdown guide for the 13-image demo catalog."""
    if lang == "ar":
        return (
            "### بيئة تجربة فورية بدون رفع ملفات (13 صورة أثاث حقيقية)\n"
            "- **✅ 4 مجموعات تكرار يجب دمجها:** نسخة مطابقة بالبايت، كرسي جلد مقصوص/مختوم، كنبة صفراء بزاويتين في غرفة معيشة، وكنبة كحلي بزوايا استوديو.\n"
            "- **🛡️ مصيدتا ألوان يجب رفضهما (Color Gate):** الكنبة الكحلي مقابل نسخة **حمراء**، والكنبة الصفراء مقابل نسخة **تركواز** بنفس الغرفة.\n"
            "- **🔍 3 منتجات فريدة:** تبقى منفصلة بالكامل."
        )
    return (
        "### Zero-Setup Reviewer Sandbox (13 Curated Furniture Images)\n"
        "- **✅ 4 True Duplicate Groups (Merged):** Exact byte copy (Bouclé Sofa), cropped+watermarked Leather Chair, cross-angle living-room Mustard Sofa, and studio-angled Navy Velvet Sofa.\n"
        "- **🛡️ 2 Color-Variant Traps (Blocked by Color Gate):** Navy Sofa vs **Crimson Red** variant, and Mustard Room Sofa vs **Teal** variant (CLIP > 92%, stopped by CIE-Lab ΔE).\n"
        "- **🔍 3 Unique Distractors:** Grey Armchair, Walnut Coffee Table, Black Oak Table stay strictly ungrouped."
    )


def _benchmarks_md(lang: str) -> str:
    """Build audited benchmark comparison Markdown for Tab 3."""
    if lang == "ar":
        return f"""### نتائج التحقق التجريبي الموثقة على Tesla T4 GPU
مرجع الكود والنتائج الكاملة على Kaggle: [Interactive GPU Benchmark Notebook (v4)]({KAGGLE_NOTEBOOK_URL})

| المقياس / السيناريو | IKEA-FS (الأساس v1) | IKEA-FS (المطوّر v4) | RoyalOak (المطوّر v4) |
|---|---|---|---|
| **حجم الكتالوج وحالات التقييم** | 1,141 صورة (48 حالة مرجعية) | 1,141 صورة (48 حالة مرجعية) | 671 صورة (623 منتج + 48 حالة) |
| **الدقة الإجمالية (48 حالة اختبار)** | 85.4% (41 / 48) | **95.8% (46 / 48)** | **100.0% (48 / 48)** |
| **النسخ المتطابقة بالبايت** | 12 / 12 (100%) | **12 / 12 (100%)** | **12 / 12 (100%)** |
| **قص 4% + تصغير + JPEG q70** | 5 / 12 (41.7%) | **10 / 12 (83.3%)** | **12 / 12 (100%)** |
| **صور بعلامة مائية (Watermark)** | 12 / 12 (100%) | **12 / 12 (100%)** | **12 / 12 (100%)** |
| **منع مصائد الألوان + نقاء الفئات** | 12 / 12 مرجعية | **12 / 12 + فصل ألوان 100%** | **12 / 12 + 98.6% نقاء فئات** |
| **زمن المعالجة (Tesla T4 GPU)** | 643 ثانية | 620.9 ثانية | 103.6 ثانية |

> **ملاحظة تدقيق البيانات:** مجموعة RoyalOak الأصلية احتوت على **678 لافتة موقع `(425×210)`** مكررة حتى 74 مرة في قوائم الموقع؛ تمت تصفيتها لعزل صور المنتجات الفعلية `(480×300)` قبل التقييم.
"""
    return f"""### Audited Multi-Dataset GPU Benchmarks (Tesla T4)
Reproducible Kaggle Pipeline: [Interactive GPU Benchmark Notebook (v4)]({KAGGLE_NOTEBOOK_URL})

| Metric / Stress Scenario | IKEA-FS (Baseline v1) | IKEA-FS (Upgraded v4) | RoyalOak Furniture (Upgraded v4) |
|---|---|---|---|
| **Catalog + Ground Truth Scope** | 1,141 images (1,093 IKEA + 48 GT) | 1,141 images (1,093 IKEA + 48 GT) | 671 images (623 clean products + 48 GT) |
| **Overall Accuracy (48 pairs)** | 85.4% (41 / 48) | **95.8% (46 / 48)** | **100.0% (48 / 48)** |
| **Exact Byte Copies** | 12 / 12 (100.0%) | **12 / 12 (100.0%)** | **12 / 12 (100.0%)** |
| **4% Crop + 80% Resize + JPEG q70** | 5 / 12 (41.7%) | **10 / 12 (83.3%)** | **12 / 12 (100.0%)** |
| **Watermarked Copies** | 12 / 12 (100.0%) | **12 / 12 (100.0%)** | **12 / 12 (100.0%)** |
| **Color-Variant Rejection & Purity** | 12 / 12 GT | **12 / 12 GT + 100% color split** | **12 / 12 GT + 98.6% category purity** |
| **Runtime (Tesla T4 GPU)** | 643s (~10.7 min) | 620.9s (~10.3 min) | 103.6s (~1.7 min) |

> **Dataset Integrity Audit:** Raw `patilaakash619/royaloak` folders contained **678 repeated navigation banners** `(425×210)` across HTML category dumps; filtering banner dimensions isolated 623 true `(480×300)` furniture photographs before evaluation.
"""


def build_demo() -> gr.Blocks:
    """Construct the Gradio 6 interactive Blocks application."""
    default_cfg = _load_default_config()
    demo_gallery_data = _list_demo_images()

    with gr.Blocks(title="Furniture Duplicate Detection AI", delete_cache=(3600, 3600)) as demo:
        with gr.Row():
            lang_state = gr.Radio(
                choices=[("🇬🇧 English", "en"), ("🇪🇬 العربية", "ar")],
                value="en",
                label="Language / اللغة",
                scale=1,
            )

        hero_md = gr.HTML(value=_hero_html("en"))

        with gr.Accordion(
            "⚙️ Pipeline & Color Gate Thresholds (Loaded from config.yaml)", open=False
        ):
            with gr.Row():
                color_slider = gr.Slider(
                    minimum=5.0,
                    maximum=40.0,
                    value=float(default_cfg.color_distance_threshold),
                    step=0.5,
                    label="CIE-Lab Color Gate Limit (ΔE) — lower = stricter color separation",
                )
                sim_slider = gr.Slider(
                    minimum=0.85,
                    maximum=0.99,
                    value=float(default_cfg.similarity_threshold),
                    step=0.005,
                    label="OpenCLIP ViT-B/32 Cosine Similarity Threshold",
                )
                phash_slider = gr.Slider(
                    minimum=2,
                    maximum=14,
                    value=int(default_cfg.phash_threshold),
                    step=1,
                    label="Perceptual Hash (pHash) Max Hamming Distance",
                )
                bg_checkbox = gr.Checkbox(
                    value=bool(default_cfg.use_background_removal),
                    label="Foreground Segmentation (rembg u2netp)",
                )

        with gr.Tabs():
            with gr.Tab("🎯 1-Click Live Demo (13 Images)"):
                demo_info = gr.Markdown(value=_demo_info_md("en"))
                with gr.Row():
                    run_demo_btn = gr.Button(
                        "🚀 Run 6-Stage AI Pipeline (Instant CPU • Unlimited)",
                        variant="primary",
                        size="lg",
                        scale=3,
                    )
                    run_demo_gpu_btn = gr.Button(
                        "⚡ Run on ZeroGPU (A10G)",
                        variant="secondary",
                        size="lg",
                        scale=1,
                    )
                with gr.Accordion(
                    f"👀 Preview All {len(demo_gallery_data)} Built-in Demo Images", open=False
                ):
                    gr.Gallery(
                        value=demo_gallery_data,
                        columns=5,
                        object_fit="contain",
                        height="auto",
                        show_label=False,
                    )

            with gr.Tab("📤 Upload Your Own Catalog (.zip / Images)"):
                upload_note = gr.Markdown(
                    "Upload a **.zip archive** or **multiple product images** (`.jpg`, `.png`, `.webp`) to test your own furniture catalog."
                )
                uploader = gr.File(
                    label="Upload Images or .zip Archive",
                    file_count="multiple",
                    file_types=["image", ".zip"],
                )
                with gr.Row():
                    run_upload_btn = gr.Button(
                        "🔍 Deduplicate Uploaded Catalog (Instant CPU • Unlimited)",
                        variant="primary",
                        size="lg",
                        scale=3,
                    )
                    run_upload_gpu_btn = gr.Button(
                        "⚡ Deduplicate on ZeroGPU (A10G)",
                        variant="secondary",
                        size="lg",
                        scale=1,
                    )

            with gr.Tab("📊 Audited Kaggle GPU Benchmarks"):
                bench_md = gr.Markdown(value=_benchmarks_md("en"))

        gr.Markdown("---")
        summary_html_out = gr.HTML(
            value="<div class='stage-funnel'>Click <b>Run 6-Stage AI Pipeline on Built-in Demo Catalog</b> above to run live inference and inspect results.</div>"
        )

        guardrail_title = gr.Markdown(
            "### 🛡️ Mandatory Color Gate Guardrail — Prevented False Positives"
        )
        guardrail_html_out = gr.HTML(value="")
        guardrail_gallery_out = gr.Gallery(
            label="Side-by-Side Blocked Color-Variant Traps",
            columns=4,
            object_fit="contain",
            height="auto",
        )

        groups_title = gr.Markdown("### 👁️ Visual Duplicate Groups Inspector")
        groups_gallery_out = gr.Gallery(
            label="Detected & Verified Duplicate Groups",
            columns=4,
            object_fit="contain",
            height="auto",
        )

        table_title = gr.Markdown("### 📋 Detailed Group Report & Exports")
        table_out = gr.Dataframe(interactive=False, wrap=True)

        with gr.Row():
            xlsx_out = gr.File(label="📥 Filtered Duplicates (.xlsx)", interactive=False)
            csv_out = gr.File(label="📥 Filtered Duplicates (.csv)", interactive=False)
            inv_out = gr.File(label="📥 Master All-Images Inventory (.xlsx)", interactive=False)

        def _switch_language(
            lang: str,
        ) -> tuple[str, str, str, str, str, str, str, str, str, str, str]:
            return (
                _hero_html(lang),
                _demo_info_md(lang),
                _tr(
                    lang,
                    "🚀 Run 6-Stage AI Pipeline (Instant CPU • Unlimited)",
                    "🚀 تشغيل خط المعالجة (6 مراحل — تشغيل فوري بلا حد)",
                ),
                _tr(
                    lang,
                    "⚡ Run on ZeroGPU (A10G)",
                    "⚡ تشغيل عبر ZeroGPU (A10G)",
                ),
                _tr(
                    lang,
                    "Upload a **.zip archive** or **multiple product images** (`.jpg`, `.png`, `.webp`) to test your own furniture catalog.",
                    "ارفع **ملف ZIP مضغوط** أو **مجموعة صور منتجات** لاختبار الأداة على صورك الخاصة.",
                ),
                _tr(
                    lang,
                    "🔍 Deduplicate Uploaded Catalog (Instant CPU • Unlimited)",
                    "🔍 فحص الصور المرفوعة (تشغيل فوري بلا حد)",
                ),
                _tr(
                    lang,
                    "⚡ Deduplicate on ZeroGPU (A10G)",
                    "⚡ فحص المرفوعات على ZeroGPU (A10G)",
                ),
                _benchmarks_md(lang),
                _tr(
                    lang,
                    "### 🛡️ Mandatory Color Gate Guardrail — Prevented False Positives",
                    "### 🛡️ بوابة التحقق اللوني — أزواج متشابهة شكلياً ومرفوضة لاختلاف اللون",
                ),
                _tr(
                    lang,
                    "### 👁️ Visual Duplicate Groups Inspector",
                    "### 👁️ معرض الفحص البصري للمجموعات المتطابقة",
                ),
                _tr(
                    lang,
                    "### 📋 Detailed Group Report & Exports",
                    "### 📋 جدول النتائج وتنزيل التقارير",
                ),
            )

        lang_state.change(
            fn=_switch_language,
            inputs=[lang_state],
            outputs=[
                hero_md,
                demo_info,
                run_demo_btn,
                run_demo_gpu_btn,
                upload_note,
                run_upload_btn,
                run_upload_gpu_btn,
                bench_md,
                guardrail_title,
                groups_title,
                table_title,
            ],
        )

        result_outputs = [
            summary_html_out,
            guardrail_html_out,
            guardrail_gallery_out,
            groups_gallery_out,
            table_out,
            xlsx_out,
            csv_out,
            inv_out,
        ]

        run_demo_btn.click(
            fn=run_builtin_demo,
            inputs=[bg_checkbox, color_slider, sim_slider, phash_slider, lang_state],
            outputs=result_outputs,
        )
        run_demo_gpu_btn.click(
            fn=run_builtin_demo_gpu,
            inputs=[bg_checkbox, color_slider, sim_slider, phash_slider, lang_state],
            outputs=result_outputs,
        )
        run_upload_btn.click(
            fn=run_uploaded_catalog,
            inputs=[uploader, bg_checkbox, color_slider, sim_slider, phash_slider, lang_state],
            outputs=result_outputs,
        )
        run_upload_gpu_btn.click(
            fn=run_uploaded_catalog_gpu,
            inputs=[uploader, bg_checkbox, color_slider, sim_slider, phash_slider, lang_state],
            outputs=result_outputs,
        )

    return demo


demo = build_demo()

if __name__ == "__main__":
    _warm_caches_safely()
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"),
        css=CUSTOM_CSS,
    )
