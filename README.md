# MindfulMoment: Meditation Recommender App

MindfulMoment is a simple, customizable Streamlit app that recommends a meditation technique based on your current mental or emotional state. It combines traditional ML classification, semantic similarity using sentence embeddings, and an integration with TinyLLaMA to generate step-by-step guided instructions.

---

## Features

- **Text-based mental state input** (e.g., "I feel anxious and unfocused")
- Two recommendation modes:
  - **Simple ML Classifier** (TF-IDF + Logistic Regression)
  - **Semantic Similarity** using SentenceTransformer
- Meditation types include:
  - Breath Awareness  
  - Body Scan  
  - Loving Kindness  
  - Mantra Meditation  
  - Walking Meditation  
  - Vipassana Noting  
  - Tibetan Visualization  
  - Zen Koan Reflection
- Uses TinyLLaMA (via [Ollama](https://ollama.ai)) to generate long-form guided scripts
- Customizable meditation reminder timer every X minutes


