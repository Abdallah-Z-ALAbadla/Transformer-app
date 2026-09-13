# -*- coding: utf-8 -*-
"""
model_utils.py
==============
كل ما يتعلق بالنموذج نفسه: تحميله، التوكنيزيشن، التوليد، واستخراج أوزان الانتباه.
هذا الملف لا يحتوي على أي كود Streamlit — فقط منطق النموذج (Model Logic)،
بحيث يمكن استخدامه أو اختباره بشكل مستقل عن الواجهة.

النموذج المستخدم: Qwen2.5-1.5B-Instruct
    - نموذج أقوى بكثير من gpt2 وقادر فعلياً على الحوار (Instruction-Tuned)،
      وفي نفس الوقت خفيف بما يكفي ليعمل على كرت شاشة NVIDIA متوسط (حتى ~6GB VRAM).
    - إذا كان عندك VRAM أكبر (8GB فأكثر) وتريد جودة أعلى، بدّل القيمة أدناه إلى:
          "Qwen/Qwen2.5-3B-Instruct"   أو   "microsoft/Phi-3-mini-4k-instruct"
      بدون أي تغيير آخر في الكود — كل الدوال هنا مكتوبة بشكل عام (Generic)
      وتتعامل مع أي نموذج Instruct من هذه العائلات.
"""

import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# اسم النموذج الجاهز (Pre-trained) الذي سنستخدمه
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# استخدام GPU تلقائياً إن كان متوفراً، وإلا الرجوع إلى CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """
    تحميل النموذج والـ Tokenizer الجاهزين على الجهاز المتوفر (GPU إن وُجد).
    نستخدم attn_implementation="eager" عمداً، لأن هذا هو التنفيذ الوحيد الذي
    يُرجع أوزان الانتباه (Attention Weights) كاملة عند output_attentions=True؛
    التنفيذات الأسرع (SDPA / FlashAttention) لا تُرجعها.
    (تُستدعى هذه الدالة مرة واحدة فقط من app.py عبر st.cache_resource)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        attn_implementation="eager",
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def get_model_info(model):
    """
    استخراج معلومات بنية النموذج (Config) الحقيقية.
    مكتوبة بشكل عام لأن أسماء الحقول تختلف بين عائلات النماذج:
        - GPT-2: n_layer, n_head, n_embd
        - Qwen / LLaMA / Phi: num_hidden_layers, num_attention_heads, hidden_size
    """
    config = model.config
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))
    num_heads = getattr(config, "num_attention_heads", getattr(config, "n_head", None))
    embed_dim = getattr(config, "hidden_size", getattr(config, "n_embd", None))
    vocab_size = config.vocab_size
    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "embed_dim": embed_dim,
        "vocab_size": vocab_size,
    }


def tokenize_text(tokenizer, text: str):
    """
    تحويل النص إلى Tokens و Token IDs.
    - Token: أصغر وحدة نصية يقسّم النموذج النص إليها.
    - Token ID: رقم صحيح فريد يمثل كل Token داخل قاموس النموذج (Vocabulary).
    """
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return tokens, input_ids


def build_chat_messages(history, max_turns: int = 6):
    """
    بناء قائمة رسائل بصيغة الحوار المعيارية (Chat Format) من آخر عدة أدوار:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    النماذج المضبوطة على التعليمات (Instruction-Tuned) مثل Qwen تتوقع هذا الشكل
    بالضبط (وليس نص خام "User: ... AI: ...")، ويُستخدم مع Chat Template الخاص
    بالنموذج لتوليد الرد. هذا هو ما يفعّل "ذاكرة المحادثة": كل الأدوار السابقة
    تُعاد كسياق مع كل طلب توليد جديد.
    """
    recent = history[-(max_turns * 2):]
    messages = []
    for turn in recent:
        role = "user" if turn["role"] == "user" else "assistant"
        messages.append({"role": role, "content": turn["text"]})
    return messages


def generate_response(tokenizer, model, messages: list, temperature: float,
                       top_k: int, max_new_tokens: int):
    """
    توليد رد كامل + عرض توزيع احتمالات التوكن التالي (Next-Token Prediction)
    + استخراج أوزان الانتباه الذاتي (Self-Attention) على النص الناتج.
    """
    # نحوّل قائمة الرسائل إلى نص مُهيأ بصيغة الحوار الخاصة بالنموذج (Chat Template).
    # نستخدم return_dict=True صراحةً حتى نحصل دائماً على نفس الشكل (BatchEncoding يحوي
    # input_ids و attention_mask)، بغض النظر عن إصدار مكتبة transformers المثبّت عندك.
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    # --- توزيع احتمالات التوكن التالي عند أول خطوة توليد (لأغراض توضيحية) ---
    with torch.no_grad():
        first_step_out = model(input_ids, attention_mask=attention_mask)
        logits = first_step_out.logits[0, -1, :].float()
        scaled_logits = logits / max(temperature, 1e-5)
        probs = torch.softmax(scaled_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=min(10, model.config.vocab_size))
        next_token_table = pd.DataFrame({
            "التوكن (Token)": [tokenizer.decode([tid]) for tid in top_ids],
            "Token ID": top_ids.tolist(),
            "الاحتمال (Probability)": [round(p.item(), 4) for p in top_probs],
        })

    # --- التوليد الفعلي للرد الكامل ---
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time

    generated_ids = output[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not generated_text:
        generated_text = "(لم يتمكن النموذج من توليد رد واضح، جرّب تعديل المعاملات)"

    num_new_tokens = len(generated_ids)

    # --- أوزان الانتباه الذاتي على النص الكامل بعد التوليد ---
    full_ids = output[:, : input_ids.shape[1] + num_new_tokens]
    with torch.no_grad():
        attn_out = model(full_ids, output_attentions=True)
    attentions = [a.float().cpu() for a in attn_out.attentions]
    full_tokens = tokenizer.convert_ids_to_tokens(full_ids[0])

    return {
        "text": generated_text,
        "elapsed": elapsed,
        "num_new_tokens": num_new_tokens,
        "next_token_table": next_token_table,
        "attentions": attentions,
        "tokens": full_tokens,
    }