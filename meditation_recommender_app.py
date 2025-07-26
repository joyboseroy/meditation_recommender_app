# meditation_recommender_app.py

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import subprocess
import time

# ---------- Training Data ----------
data = pd.DataFrame({
    "text": [
        "I feel anxious and can’t focus",
        "I am physically exhausted",
        "I want to cultivate compassion",
        "I feel restless and distracted",
        "I want to reflect on a mantra",
        "I need to calm my breath",
        "My thoughts are racing",
        "I feel tired and sore",
        "I feel lonely but grateful",
        "I want to walk and breathe",
        "I want to sit silently and reflect on a riddle",
        "I want to observe my thoughts and feelings as they arise",
        "I want to visualize a deity from Tibetan Buddhism"
    ],
    "meditation": [
        "Breath Awareness",
        "Body Scan",
        "Loving Kindness",
        "Breath Awareness",
        "Mantra Meditation",
        "Breath Awareness",
        "Breath Awareness",
        "Body Scan",
        "Loving Kindness",
        "Walking Meditation",
        "Zen Koan Reflection",
        "Vipassana Noting",
        "Tibetan Visualization"
    ]
})

# ---------- ML Classifier ----------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])
pipeline.fit(data["text"], data["meditation"])

# ---------- Embedding Setup ----------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
prototypes = {
    "Breath Awareness": "Meditation that focuses on anchoring attention to breath.",
    "Body Scan": "Progressive relaxation and bodily awareness.",
    "Loving Kindness": "Sending kind thoughts to self and others.",
    "Mantra Meditation": "Repeating a sacred phrase or sound.",
    "Walking Meditation": "Meditative walking with attention to steps and breath.",
    "Vipassana Noting": "Meditation noting sensations and thoughts as they arise.",
    "Tibetan Visualization": "Visualizing a deity such as Buddha Shakyamuni to receive blessings.",
    "Zen Koan Reflection": "Silent meditation focused on a Zen riddle or paradox."
}
proto_embeds = {k: embed_model.encode(v) for k, v in prototypes.items()}

# ---------- Load Prompt File ----------
guide_df = pd.DataFrame({
    "meditation": [
        "Breath Awareness",
        "Body Scan",
        "Loving Kindness",
        "Mantra Meditation",
        "Walking Meditation",
        "Vipassana Noting",
        "Tibetan Visualization",
        "Zen Koan Reflection"
    ],
    "guide_prompt": [
        "Be mindful of the in-breath as it enters through the nose, expands the belly, and the out-breath as it leaves.",
        "Scan your body from head to toe, pausing at each part to notice sensation, tightness, or warmth.",
        "Start by cultivating love for yourself, then extend that to loved ones, strangers, and all beings.",
        "Silently repeat a mantra like 'Om Mani Padme Hum', letting it flow with your breath.",
        "Walk slowly and mindfully, noticing each step as you lift, move, and place your foot.",
        "Gently note the arising of sensations or thoughts, like 'pain', 'itch', or 'thinking', without judgment.",
        "Visualize Buddha Shakyamuni in front of you, radiant and golden, and imagine receiving light and blessings.",
        "Sit with a Zen question like 'What is your original face before you were born?' without trying to answer."
    ]
})

# ---------- TinyLLaMA Helper ----------
def query_tinyllama(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "tinyllama", prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout.strip()
    except Exception as e:
        return f"(Failed to fetch guide from TinyLLaMA: {e})"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="MindfulMoment")
st.title("MindfulMoment - Meditation Recommender")

with st.form("mood_form"):
    user_input = st.text_area("Describe your current mental state:", "I feel anxious and unfocused")
    use_mode = st.radio("Select recommendation mode:", ["Simple ML", "Semantic Similarity"])
    remind_every = st.slider("Remind me to meditate every X minutes:", 5, 60, 15)
    submitted = st.form_submit_button("Get Recommendation")

if submitted:
    st.subheader("Recommendation")
    meditation_type = None
    if use_mode == "Simple ML":
        prediction = pipeline.predict([user_input])[0]
        meditation_type = prediction
        st.success(f"Try **{meditation_type}** meditation.")
    else:
        input_vec = embed_model.encode(user_input)
        best = max(proto_embeds.items(), key=lambda x: cosine_similarity([input_vec], [x[1]])[0][0])
        meditation_type = best[0]
        st.success(f"Closest match: **{meditation_type}** meditation.")

    guide_row = guide_df[guide_df["meditation"].str.strip().str.lower() == meditation_type.lower()]
    if not guide_row.empty:
        prompt = guide_row.iloc[0]["guide_prompt"] + " Provide a longer, step-by-step instruction, suitable for a 5-minute guided practice."
        response = query_tinyllama(prompt)

        if "Failed to fetch" in response:
            st.error(f"LLM Error: {response}")
        else:
            st.markdown("#### TinyLLaMA Guide:")
            st.info(response)

# ---------- Timer Reminder ----------
if "last_reminder" not in st.session_state:
    st.session_state.last_reminder = time.time()

elapsed = (time.time() - st.session_state.last_reminder) / 60
if elapsed > remind_every:
    st.warning("Time to take a mindful break!")
    st.session_state.last_reminder = time.time()

st.info("Tip: Meditation doesn't have to be long. Even 2 minutes of mindful breathing can help reset.")
