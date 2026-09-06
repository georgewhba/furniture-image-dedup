@echo off
chcp 65001 > nul
title واجهة الويب الذكية لفرز وتطابق صور الأثاث

echo ====================================================================
echo        جاري تشغيل واجهة الويب التفاعلية (Streamlit Web UI)...
echo ====================================================================

cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [خطأ] لم يتم العثور على بايثون مثبت في جهازك!
    pause
    exit /b 1
)

if not exist ".venv\Scripts\activate.bat" (
    echo [تنبيه] إعداد البيئة الافتراضية لأول مرة...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

echo [نجاح] جاري فتح واجهة النظام في المتصفح...
streamlit run streamlit_app.py

pause
