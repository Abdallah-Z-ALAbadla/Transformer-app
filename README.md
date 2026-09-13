# تشغيل تطبيق الشات بوت

## التشغيل محلياً (الأسهل والأنسب لهذا المشروع)
```bash
pip install -r requirements.txt
streamlit run app.py
```
سيفتح المتصفح تلقائياً على الرابط المحلي.

## التشغيل من Google Colab
تطبيقات Streamlit تحتاج خادماً يعمل باستمرار، وColab لا يدعم فتح منفذ محلي مباشرة،
لذلك يلزم استخدام أداة نفق مثل `localtunnel`. نفّذ التالي في خلية واحدة داخل Colab:

```python
!pip install streamlit transformers torch matplotlib pandas -q
!wget -q -O app.py https://<رفع الملف على GitHub أو رابط مباشر>
!npm install -g localtunnel -q

import subprocess, time
subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])
time.sleep(5)
!npx localtunnel --port 8501
```
سيظهر رابط عام (URL) يمكن فتحه من المتصفح للوصول إلى التطبيق.

## ملاحظة حول النموذج
يستخدم التطبيق نموذج `Qwen/Qwen2.5-1.5B-Instruct` — نموذج مضبوط على التعليمات (Instruction-Tuned)
يعطي ردوداً منطقية وحوارية حقيقية (على عكس `gpt2` الذي كان يكمل الجمل فقط بدون تدريب على الحوار).

**متطلبات التشغيل:**
- كرت شاشة NVIDIA — الحد الأدنى تقريباً 4-6GB VRAM (يعمل تلقائياً على GPU إن توفر، وإلا يرجع لـ CPU).
- أول تشغيل سيحمّل ملفات النموذج من HuggingFace (حجمها تقريباً 3 جيجابايت)، فيحتاج اتصال إنترنت
  ووقتاً حسب سرعة الشبكة، ثم تُخزّن محلياً ولا يُعاد تحميلها في التشغيلات القادمة.

**تأكد أن نسخة PyTorch مثبّتة بدعم CUDA** (وليس نسخة CPU فقط)، وإلا فلن يُستخدم كرت الشاشة:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```
إذا ظهرت `False` مع أنك تملك GPU، ثبّت نسخة PyTorch المتوافقة مع CUDA من الرابط الرسمي:
https://pytorch.org/get-started/locally/

**لتجربة نموذج أقوى/أخف:** غيّر قيمة `MODEL_NAME` في أول `model_utils.py` فقط — كل الكود الباقي
عام (Generic) ويتعامل مع أي نموذج Instruct من نفس العائلة (Qwen / LLaMA / Phi) بدون أي تعديل آخر.
