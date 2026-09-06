"""Google Colab GPU Runner for Furniture Deduplication.
تشغيل واجهة Streamlit على كارت شاشة NVIDIA T4 في Google Colab مع رابط خارجي فوري.
"""

import os
import subprocess
import sys
import time

def setup_and_run():
    print("=" * 70)
    print("🚀 جاري تهيئة نظام فرز وتطابق صور الأثاث على كارت الشاشة GPU...")
    print("=" * 70)

    # 1. Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ تم اكتشاف كارت الشاشة بنجاح: {gpu_name}")
        else:
            print("⚠️ تنبيه: لم يتم تفعيل الـ GPU، يعمل حالياً على الـ CPU.")
    except Exception as e:
        print(f"خطأ في فحص Torch/CUDA: {e}")

    # 2. Install Cloudflare Tunnel helper if not present
    try:
        from pycloudflared import try_cloudflare
    except ImportError:
        print("📦 جاري تثبيت pycloudflared...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pycloudflared"])
        from pycloudflared import try_cloudflare

    # 3. Start Streamlit in background
    print("🌐 جاري تشغيل خادم واجهة الويب Streamlit...")
    streamlit_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.headless", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(4)

    # 4. Expose public tunnel
    print("🔗 جاري إنشاء الرابط السريع المباشر عبر Cloudflare...")
    tunnel = try_cloudflare(port=8501)
    
    print("\n" + "=" * 70)
    print("🎉 تم تشغيل التطبيق بنجاح بقوة الـ GPU!")
    print(f"👉 افتح هذا الرابط في المتصفح للبدء فوراً:\n\n   {tunnel.tunnel}\n")
    print("=" * 70 + "\n")

    try:
        streamlit_proc.wait()
    except KeyboardInterrupt:
        streamlit_proc.terminate()
        print("\nتم إيقاف التطبيق.")

if __name__ == "__main__":
    setup_and_run()
