"""Interactive Assistant for Furniture Deduplication System.
معالج تفاعلي سهل لتشغيل نظام فرز وتطابق صور الأثاث بضغطة زر.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging
from src.pipeline import run


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    print("=" * 70)
    print("      نظام الكشف الذكي عن صور الأثاث المتطابقة والمكررة")
    print("      Furniture Product-Image Duplicate Detection System")
    print("=" * 70)
    print(" المميزات:")
    print("  ✓ فرز التطابق الدقيق (نفس الموديل + نفس اللون فقط).")
    print("  ✓ استبعاد المنتجات المتشابهة في الشكل والمختلفة في اللون.")
    print("  ✓ لا يتم حذف أو تعديل أي صورة نهائياً (قرار بشري 100%).")
    print("  ✓ تقرير مفصل بصيغة Excel و CSV مع درجات الثقة.")
    print("=" * 70)
    print()


def get_input_directory() -> Path:
    while True:
        prompt = (
            "📌 أدخل مسار مجلد الصور (أو اسحب المجلد وأفلته هنا ثم اضغط Enter):\n"
            "📁 Folder path: "
        )
        path_str = input(prompt).strip()
        
        # Remove surrounding quotes if dragged & dropped in terminal
        path_str = path_str.strip('"').strip("'")
        
        if not path_str:
            print("⚠️ الرجاء إدخال مسار صحيح.")
            continue
            
        folder_path = Path(path_str).resolve()
        if not folder_path.exists():
            print(f"❌ المسار غير موجود: {folder_path}\nحاول مرة أخرى.")
            continue
        if not folder_path.is_dir():
            print(f"❌ المسار ليس مجلداً: {folder_path}\nحاول مرة أخرى.")
            continue
            
        return folder_path


def get_sample_choice() -> int | None:
    print("\nاختر وضع التشغيل:")
    print("  [1] فحص عينة تجريبية أولى (500 صورة)")
    print("  [2] فحص عينة تجريبية أولى (1000 صورة)")
    print("  [3] فحص المكتبة بالكامل (كافة الصور الموجودة)")
    print("  [4] تحديد رقم مخصص للصور")
    
    while True:
        choice = input("👉 اختيارك (1-4) [الافتراضي: 3]: ").strip()
        if choice in ("", "3"):
            return None
        elif choice == "1":
            return 500
        elif choice == "2":
            return 1000
        elif choice == "4":
            num_str = input("أدخل عدد الصور المطلوب فحصها: ").strip()
            if num_str.isdigit() and int(num_str) > 0:
                return int(num_str)
            print("رقم غير صحيح، سيتم فحص كافة الصور.")
            return None
        else:
            print("⚠️ اختيار غير صحيح، حاول ثانية.")


def open_report_file(file_path: Path):
    """Open the generated report or its enclosing folder in Windows Explorer."""
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(file_path)], check=False)
        else:
            subprocess.run(["xdg-open", str(file_path)], check=False)
    except Exception as exc:
        print(f"يمكنك فتح الملف يدوياً من: {file_path}")


def main():
    clear_screen()
    print_banner()
    
    # 1. Input directory
    input_dir = get_input_directory()
    print(f"✅ تم اختيار المجلد: {input_dir}\n")
    
    # 2. Sample size
    sample_size = get_sample_choice()
    if sample_size:
        print(f"✅ سيتم فحص أول {sample_size} صورة كعينة تجريبية.\n")
    else:
        print("✅ سيتم فحص كافة الصور الموجودة في المجلد.\n")
        
    # 3. Output path
    default_output = input_dir / "تقرير_تطابق_الصور.xlsx"
    print(f"📊 مسار حفظ التقرير الافتراضي:\n  {default_output}")
    custom_out = input("هل ترغب في تغيير اسم أو مسار التقرير؟ (اضغط Enter للمتابعة بالمسار الافتراضي): ").strip()
    if custom_out:
        custom_out = custom_out.strip('"').strip("'")
        output_path = Path(custom_out).resolve()
        if output_path.is_dir():
            output_path = output_path / "تقرير_تطابق_الصور.xlsx"
    else:
        output_path = default_output
        
    # 4. Config & cache
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        print(f"❌ لم يتم العثور على ملف الإعدادات: {config_path}")
        input("اضغط Enter للخروج...")
        return 1
        
    config = load_config(config_path)
    setup_logging(config)
    cache_dir = PROJECT_ROOT / ".dedup_cache"
    
    print("\n" + "=" * 70)
    print("🚀 جاري بدء الفرز الذكي الآن...")
    print("=" * 70)
    
    try:
        groups = run(
            input_dir=input_dir,
            output_path=output_path,
            config=config,
            cache_dir=cache_dir,
            sample_size=sample_size,
            resume=True,
            verbose=False,
        )
        
        print("\n" + "=" * 70)
        print("🎉 تم الانتهاء بنجاح!")
        print(f"📊 عدد مجموعات التكرار المؤكدة: {len(groups)}")
        print(f"📁 تم تصدير تقرير الإكسل: {output_path}")
        print(f"📁 تم تصدير تقرير CSV: {output_path.with_suffix('.csv')}")
        print("=" * 70)
        
        # Ask to open
        open_choice = input("\nهل تريد فتح تقرير الإكسل الآن؟ (Y/n) [الافتراضي: نعم]: ").strip().lower()
        if open_choice in ("", "y", "yes", "نعم"):
            open_report_file(output_path)
            
    except KeyboardInterrupt:
        print("\n⚠️ تم إيقاف العملية من قبل المستخدم. تم حفظ التقدم في الكاش.")
    except Exception as exc:
        print(f"\n❌ حدث خطأ أثناء المعالجة: {exc}")
        import traceback
        traceback.print_exc()
        
    input("\nاضغط Enter لإغلاق النافذة...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
