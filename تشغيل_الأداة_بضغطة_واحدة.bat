@echo off
chcp 65001 > nul
title نظام فرز وتطابق صور الأثاث - Furniture Deduplication System

echo ====================================================================
echo             جاري التحقق من بيئة العمل وتشغيل الأداة...
echo ====================================================================

cd /d "%~dp0"

:: 1. Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [خطأ] لم يتم العثور على بايثون (Python) مثبت في جهازك!
    echo يرجى تحميل وتثبيت بايثون من: https://www.python.org/downloads/
    echo وتأكد من تحديد خيار "Add Python to PATH" أثناء التثبيت.
    pause
    exit /b 1
)

:: 2. Check or create Virtual Environment
if not exist ".venv\Scripts\activate.bat" (
    echo [تنبيه] جاري إعداد البيئة الافتراضية لأول مرة... يرجى الانتظار دقيقة واحدة.
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [تنبيه] جاري تثبيت المتطلبات الضرورية تلقائياً...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

:: 3. Launch the interactive wizard
python interactive_wizard.py

pause
