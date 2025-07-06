import streamlit as st
from transformers import pipeline
from utils.data_loader import load_personas, load_landmarks
from utils.prompt_formatter import format_prompt

# Cache the model loader
def load_generator(model_path: str):
    return pipeline(
        "text-generation",
        model=model_path,
        device_map="auto",
        trust_remote_code=True
    )

@st.cache_resource
# Load persona and landmark metadata
personas = load_personas("data/personas.json")
landmarks = load_landmarks("data/landmarks.json")

def main():
    """Run the Streamlit app for Vienna AI Historical Guide System."""
    st.set_page_config(page_title="Vienna AI Guide", layout="centered")
    st.sidebar.title("Select Guide & Landmark")

    persona_key = st.sidebar.selectbox("Historical Guide", list(personas.keys()))
    landmark_key = st.sidebar.selectbox("Landmark", list(landmarks.keys()))

    generator = load_generator("./checkpoints")

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    st.title("Vienna AI Historical Guide")
    user_input = st.text_input("Ask your question:")

    if user_input:
        # Append user message
        st.session_state.history.append(("user", user_input))

        # Build prompt and generate response
        prompt = format_prompt(
            persona=personas[persona_key],
            landmark=landmarks[landmark_key],
            user_query=user_input
        )
        with st.spinner("Generating response..."):
            response = generator(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                top_p=0.9
            )[0]["generated_text"]
        reply = response.replace(prompt, "").strip()

        # Append persona response
        st.session_state.history.append((persona_key, reply))

    # Render chat messages
    for speaker, message in st.session_state.history:
        if speaker == "user":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)

if __name__ == "__main__":
    main()
