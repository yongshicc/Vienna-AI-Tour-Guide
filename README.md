# Vienna-AI-Tour-Guide

# Vienna AI Historical Guide System

The **Vienna AI Historical Guide System** allows users to engage in interactive, conversational tours of Vienna’s most iconic landmarks through the voices of historical figures. By fine‑tuning a large language model with persona‑specific dialogue examples, each guide (Mozart, Empress Sisi, Freud, and Klimt) offers immersive, personalized narratives that blend historical facts with characterful storytelling.

## Key Features
- **Multiple Historical Guides**: Chat with Mozart, Empress Elisabeth (“Sisi”), Sigmund Freud, or Gustav Klimt.
- **Landmark Pairings**: Each persona is paired with two Vienna sites (e.g., Mozart Museum & Haus der Musik).
- **Fine‑Tuned Model**: Uses LoRA adapters on a state‑of‑the‑art foundation model for coherent persona voices.
- **Interactive Web UI**: Built with Streamlit for easy deployment and user‑friendly controls.
- **Extensible Data**: Personas, landmarks, and dialogue examples are stored in JSON/JSONL for easy customization.

## Architecture Overview
1. **Data Layer**: `data/` directory containing:
   - `personas.json`: Metadata and style cues for each historical figure.
   - `landmarks.json`: Descriptions and images of Vienna landmarks.
   - `persona_dialogues.jsonl`: Sample Q&A pairs for fine‑tuning.
2. **Model Fine‑Tuning**: `finetune.py` uses Hugging Face Transformers and PEFT (LoRA) to adapt the base model on persona dialogue data.
3. **Application Layer**: `app.py` hosts a Streamlit app that:
   - Loads personas and landmarks.
   - Generates dynamic prompts.
   - Streams persona responses in a chat interface.
4. **Utilities**: Helper modules in `utils/` for data loading and prompt formatting.
