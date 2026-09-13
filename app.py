# -*- coding: utf-8 -*-
"""
app.py
======
تطبيق شات بوت ذكي مبني على نموذج Transformer جاهز (Pre-trained)
Building an Intelligent Transformer-Based Chatbot

هذا الملف يحتوي فقط على واجهة Streamlit (UI).
منطق النموذج (تحميل، توليد، انتباه) موجود في model_utils.py
ودوال الرسم موجودة في ui_components.py

طريقة التشغيل:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from model_utils import (
    MODEL_NAME,
    DEVICE,
    load_model_and_tokenizer,
    get_model_info,
    tokenize_text,
    build_chat_messages,
    generate_response,
)
from ui_components import plot_attention_heatmap

# ============================================================
# إعدادات عامة للصفحة
# ============================================================
st.set_page_config(page_title="Transformer Chatbot", page_icon="🤖", layout="wide")


# ============================================================
# تحميل النموذج (مع تخزين مؤقت لتسريع التطبيق)
# ============================================================
@st.cache_resource(show_spinner="جاري تحميل نموذج الـ Transformer ...")
def get_cached_model():
    return load_model_and_tokenizer(MODEL_NAME)


tokenizer, model = get_cached_model()
model_info = get_model_info(model)
NUM_LAYERS = model_info["num_layers"]
NUM_HEADS = model_info["num_heads"]
EMBED_DIM = model_info["embed_dim"]
VOCAB_SIZE = model_info["vocab_size"]


# ============================================================
# تهيئة حالة الجلسة (Session State)
# ============================================================
def init_state():
    defaults = {
        "history": [],          # [{"role": "user"/"ai", "text": "..."}]
        "num_questions": 0,
        "total_generated_tokens": 0,
        "response_times": [],
        "last_input_ids": None,
        "last_tokens": None,
        "last_attentions": None,
        "last_next_token_probs": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


# ============================================================
# الشريط الجانبي: معلومات النموذج + معاملات التوليد + لوحة التحكم
# ============================================================
with st.sidebar:
    st.header("🧠 معلومات النموذج (Model Info)")
    st.markdown(f"""
    - **اسم النموذج:** `{MODEL_NAME}`
    - **جهاز التشغيل:** `{DEVICE.upper()}`
    - **عدد طبقات الـ Transformer:** {NUM_LAYERS}
    - **عدد رؤوس الانتباه (Attention Heads):** {NUM_HEADS}
    - **بُعد التمثيل (Embedding Dimension):** {EMBED_DIM}
    - **حجم القاموس (Vocabulary Size):** {VOCAB_SIZE:,}
    """)

    st.divider()
    st.header("⚙️ معاملات التوليد (Generation Settings)")
    temperature = st.slider("Temperature (درجة العشوائية)", 0.1, 2.0, 0.8, 0.1,
                             help="قيمة منخفضة → ردود أكثر ثباتاً ومنطقية. قيمة عالية → ردود أكثر تنوعاً وعشوائية.")
    top_k = st.slider("Top-K", 1, 100, 40, 1,
                       help="عدد أفضل الاحتمالات التي يُسمح للنموذج بالاختيار من بينها في كل خطوة توليد.")
    max_new_tokens = st.slider("أقصى عدد توكنز للتوليد (Max New Tokens)", 5, 150, 40, 5)

    st.divider()
    st.header("📊 لوحة التحكم (Dashboard)")
    avg_time = (np.mean(st.session_state.response_times)
                if st.session_state.response_times else 0.0)
    st.metric("عدد الأسئلة المطروحة", st.session_state.num_questions)
    st.metric("عدد التوكنز المولّدة (إجمالي)", st.session_state.total_generated_tokens)
    st.metric("متوسط زمن الاستجابة (ثانية)", f"{avg_time:.2f}")
    st.metric("عدد أدوار المحادثة", len(st.session_state.history) // 2)
    st.caption(f"المعاملات الحالية: Temperature={temperature} | Top-K={top_k} | Max Tokens={max_new_tokens}")

    if st.button("🗑️ مسح المحادثة"):
        st.session_state.history = []
        st.session_state.num_questions = 0
        st.session_state.total_generated_tokens = 0
        st.session_state.response_times = []
        st.rerun()


# ============================================================
# الواجهة الرئيسية
# ============================================================
st.title("🤖 شات بوت ذكي مبني على نموذج Transformer")
st.caption("مبني على نموذج GPT-2 الجاهز (Pre-trained) — لتوضيح آلية عمل الـ Transformer خطوة بخطوة")

tab_chat, tab_tokens, tab_attention, tab_predict = st.tabs(
    ["💬 المحادثة", "🔤 التوكنيزيشن", "🔍 الانتباه الذاتي", "🎯 التنبؤ بالتوكن التالي"]
)

# ------------------------------------------------------------
# تبويب المحادثة
# ------------------------------------------------------------
with tab_chat:
    for turn in st.session_state.history:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.write(turn["text"])

    user_input = st.chat_input("اكتب سؤالك هنا ...")

    if user_input:
        # 1) تسجيل رسالة المستخدم في تاريخ المحادثة (ذاكرة المحادثة)
        st.session_state.history.append({"role": "user", "text": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # 2) ترميز آخر إدخال لعرضه في تبويب التوكنيزيشن
        tokens, input_ids = tokenize_text(tokenizer, user_input)
        st.session_state.last_tokens = tokens
        st.session_state.last_input_ids = input_ids

        # 3) بناء السياق الكامل (يشمل الأدوار السابقة) وتوليد الرد
        messages = build_chat_messages(st.session_state.history)
        with st.chat_message("assistant"):
            with st.spinner("النموذج يفكّر ..."):
                result = generate_response(tokenizer, model, messages, temperature, top_k, max_new_tokens)
            st.write(result["text"])

        # 4) تحديث ذاكرة المحادثة وإحصائيات اللوحة
        st.session_state.history.append({"role": "ai", "text": result["text"]})
        st.session_state.num_questions += 1
        st.session_state.total_generated_tokens += result["num_new_tokens"]
        st.session_state.response_times.append(result["elapsed"])
        st.session_state.last_attentions = result["attentions"]
        st.session_state.last_tokens = result["tokens"]
        st.session_state.last_next_token_probs = result["next_token_table"]

        st.caption(f"⏱️ زمن الاستجابة: {result['elapsed']:.2f} ثانية | "
                   f"عدد التوكنز المولّدة: {result['num_new_tokens']}")

# ------------------------------------------------------------
# تبويب التوكنيزيشن
# ------------------------------------------------------------
with tab_tokens:
    st.subheader("العلاقة بين النص والتوكنز و Token IDs")
    st.markdown("""
    - **النص (Text):** ما يكتبه المستخدم كما هو، بلغة طبيعية.
    - **التوكن (Token):** أصغر وحدة يقسّم النموذج النص إليها؛ قد تكون كلمة كاملة، جزءاً من كلمة، أو رمز ترقيم.
    - **Token ID:** رقم صحيح فريد يمثّل كل توكن داخل قاموس النموذج (Vocabulary)، وهو ما يفهمه النموذج فعلياً وليس النص نفسه.

    التدفق: **نص → تقسيم إلى توكنز → تحويل كل توكن إلى رقم (Token ID) → تمثيل رقمي (Embedding) → معالجة داخل طبقات الـ Transformer**
    """)

    sample_text = st.text_input("جرّب كتابة أي نص لعرض تحليله:", value="مرحباً، كيف حالك؟")
    if sample_text:
        demo_tokens, demo_ids = tokenize_text(tokenizer, sample_text)
        df_tokens = pd.DataFrame({
            "#": range(1, len(demo_tokens) + 1),
            "Token": demo_tokens,
            "Token ID": demo_ids[0].tolist(),
        })
        st.dataframe(df_tokens, use_container_width=True, hide_index=True)
        st.info(f"عدد التوكنز الناتجة عن هذا النص: {len(demo_tokens)}")

    if st.session_state.last_tokens is not None and st.session_state.last_input_ids is not None:
        st.divider()
        st.caption("آخر إدخال من المستخدم في المحادثة:")
        st.dataframe(pd.DataFrame({
            "Token": tokenizer.convert_ids_to_tokens(st.session_state.last_input_ids[0]),
            "Token ID": st.session_state.last_input_ids[0].tolist(),
        }), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# تبويب الانتباه الذاتي
# ------------------------------------------------------------
with tab_attention:
    st.subheader("تصور آلية الانتباه الذاتي (Self-Attention)")
    st.markdown("""
    عند معالجة كل توكن، يحسب النموذج **وزن انتباه** تجاه كل توكن آخر في الجملة (بما فيها نفسه)،
    ليقرر أي الكلمات أكثر أهمية لفهم المعنى في هذا الموضع. الألوان الأفتح في الخريطة تعني وزن انتباه أعلى.
    """)

    if st.session_state.last_attentions is None:
        st.warning("اطرح سؤالاً أولاً من تبويب المحادثة لعرض أوزان الانتباه الخاصة به.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            layer_choice = st.slider("اختر رقم الطبقة (Layer)", 1, NUM_LAYERS, NUM_LAYERS) - 1
        with col2:
            head_choice = st.slider("اختر رقم رأس الانتباه (Head)", 1, NUM_HEADS, 1) - 1

        fig = plot_attention_heatmap(
            st.session_state.last_attentions,
            st.session_state.last_tokens,
            layer_choice,
            head_choice,
        )
        st.pyplot(fig)
        st.caption("لاحظ كيف تختلف أنماط الانتباه باختلاف الطبقة ورأس الانتباه المختارين.")

# ------------------------------------------------------------
# تبويب التنبؤ بالتوكن التالي
# ------------------------------------------------------------
with tab_predict:
    st.subheader("مفهوم Next-Token Prediction")
    st.markdown("""
    نموذج الـ Transformer التوليدي (مثل GPT-2) لا يولّد الجملة دفعة واحدة،
    بل يتنبأ في كل خطوة **بالتوكن التالي فقط** بناءً على كل ما سبقه من سياق،
    ثم يُضاف هذا التوكن إلى السياق ويُعاد التنبؤ بالتوكن الذي يليه، وهكذا حتى تكتمل الجملة.
    """)

    if st.session_state.last_next_token_probs is None:
        st.warning("اطرح سؤالاً أولاً من تبويب المحادثة لعرض توزيع احتمالات التوكن التالي.")
    else:
        st.caption("أعلى 10 توكنز مرشحة للتوكن التالي عند بداية توليد آخر رد (بعد تطبيق Temperature الحالية):")
        st.dataframe(st.session_state.last_next_token_probs, use_container_width=True, hide_index=True)
        st.bar_chart(st.session_state.last_next_token_probs.set_index("التوكن (Token)")["الاحتمال (Probability)"])
