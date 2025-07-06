import streamlit as st
import json
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Sample data for personas and landmarks
SAMPLE_PERSONAS = {
    "Mozart": {
        "name": "Wolfgang Amadeus Mozart",
        "pre_prompt": "You are Wolfgang Amadeus Mozart, the legendary 18th-century Austrian composer.",
        "traits": "lively, musical metaphors, piano gestures",
        "image": "images/mozart.png",
        "importance": "One of Austria's greatest musical prodigies whose works epitomize the Classical era in Vienna",
        "tourist_attraction": "Visitors flock to his former residence and Mozart concerts to experience his timeless compositions"
    },
    "Sisi": {
        "name": "Empress Elisabeth of Austria (Sisi)",
        "pre_prompt": "You are Empress Elisabeth of Austria, known as Sisi, famed for your poetic melancholy.",
        "traits": "poetic, longing for freedom, melancholic smile",
        "image": "images/sisi.png",
        "importance": "Beloved symbol of the Habsburg empire and Viennese society, her life reflects Austria's golden age",
        "tourist_attraction": "Tourists admire her legacy at Schönbrunn Palace and follow the 'Sisi' walking tours"
    },
    "Freud": {
        "name": "Sigmund Freud",
        "pre_prompt": "You are Sigmund Freud, the pioneering founder of psychoanalysis.",
        "traits": "analytical, subconscious focus, cigar smoking",
        "image": "images/freud.png",
        "importance": "Father of psychoanalysis whose Vienna practice shaped modern psychology worldwide",
        "tourist_attraction": "Visitors come to the Freud Museum for insight into his life, office, and groundbreaking theories"
    },
    "Klimt": {
        "name": "Gustav Klimt",
        "pre_prompt": "You are Gustav Klimt, the Viennese symbolist painter.",
        "traits": "sensory golden descriptions, brush holding",
        "image": "images/klimt.png",
        "importance": "Leader of the Viennese Secession movement, his gilded artworks define Austrian modernism",
        "tourist_attraction": "Art lovers admire his masterpieces like 'The Kiss' at the Belvedere Museum"
    }
}

SAMPLE_LANDMARKS ={
  "Schönbrunn Palace": {
    "name": "Schönbrunn Palace",
    "description": "The former summer residence of the Habsburgs, famed for its Gloriette, grand gardens, and Baroque architecture.",
    "image": "images/schonbrunn.jpg",
    "location": "Vienna, Austria"
  },
  "Sisi Museum": {
    "name": "Sisi Museum",
    "description": "Dedicated to Empress Elisabeth (Sisi), showcasing her personal artifacts, dresses, and the romantic yet tragic aspects of her life.",
    "image": "images/sisi_museum.jpg",
    "location": "Vienna, Austria"
  },
  "Mozart Museum": {
    "name": "Mozart Museum",
    "description": "Located in Mozart's former residence, displaying his original scores, instruments, and family memorabilia.",
    "image": "images/mozart_museum.jpg",
    "location": "Vienna, Austria"
  },
  "Haus der Musik": {
    "name": "Haus der Musik",
    "description": "An interactive sound museum exploring music, acoustics, and Vienna's great composers with hands‑on exhibits.",
    "image": "images/haus_der_musik.jpg",
    "location": "Vienna, Austria"
  },
  "Freud Museum": {
    "name": "Freud Museum",
    "description": "Preserves the apartment and office of Sigmund Freud, with original furniture, Freud's psychoanalytic couch, and archives.",
    "image": "images/freud_museum.jpg",
    "location": "Vienna, Austria"
  },
  "University of Vienna": {
    "name": "University of Vienna",
    "description": "One of Europe’s oldest universities where Freud studied and later taught, with historic lecture halls and archives.",
    "image": "images/university_of_vienna.jpg",
    "location": "Vienna, Austria"
  },
  "Belvedere Museum": {
    "name": "Belvedere Museum",
    "description": "A historic Baroque palace complex housing Austria's largest collection of Gustav Klimt paintings, including 'The Kiss'.",
    "image": "images/belvedere.jpg",
    "location": "Vienna, Austria"
  },
  "Kunsthistorisches Museum": {
    "name": "Kunsthistorisches Museum",
    "description": "One of the world’s most important fine arts museums, with collections spanning ancient to modern art, including masterpieces by Klimt.",
    "image": "images/kunsthistorisches.jpg",
    "location": "Vienna, Austria"
  }
}

def load_personas(file_path: str):
    """Load personas from JSON file or use sample data."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return SAMPLE_PERSONAS

def load_landmarks(file_path: str):
    """Load landmarks from JSON file or use sample data."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return SAMPLE_LANDMARKS

def format_prompt(persona, landmark, user_query):
    """Format the prompt for the AI model."""
    prompt = f"""{persona['pre_prompt']}

Personality traits: {persona['traits']}
Historical importance: {persona['importance']}
Tourist context: {persona['tourist_attraction']}

You are currently at: {landmark['name']}
Location description: {landmark['description']}
Historical context: {landmark['history']}

Please respond as {persona['name']} would, incorporating your personality traits and knowledge of both your historical significance and this location.

Tourist question: {user_query}
{persona['name']}:"""
    return prompt

@st.cache_resource
def load_simple_generator():
    """Load a simple text generator for demo purposes."""
    try:
        # Try to load a fine-tuned model if available
        if os.path.exists("./checkpoints/final_model"):
            tokenizer = AutoTokenizer.from_pretrained("./checkpoints/final_model")
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-small",
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            model = PeftModel.from_pretrained(base_model, "./checkpoints/final_model")
            return pipeline("text-generation", model=model, tokenizer=tokenizer)
        else:
            # Fallback to a simple model
            return pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None

def generate_response(generator, prompt, max_length=150):
    """Generate response using the AI model."""
    if generator is None:
        # Fallback response if model loading fails
        return "Entschuldigung, ich kann momentan nicht antworten. Das AI-Modell ist nicht verfügbar."
    
    try:
        response = generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        # Extract the generated text
        generated_text = response[0]['generated_text']
        
        # Remove the prompt from the response
        if prompt in generated_text:
            reply = generated_text.replace(prompt, "").strip()
        else:
            reply = generated_text.strip()
            
        return reply if reply else "Entschuldigung, ich konnte keine passende Antwort generieren."
        
    except Exception as e:
        return f"Fehler bei der Antwort-Generierung: {str(e)}"

def main():
    """Run the Streamlit app for Vienna AI Historical Guide System."""
    st.set_page_config(
        page_title="Vienna AI Guide", 
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Load data
    personas = load_personas("data/personas.json")
    landmarks = load_landmarks("data/landmarks.json")
    
    # Sidebar
    st.sidebar.title("🏰 Vienna AI Guide")
    st.sidebar.markdown("Choose your historical guide and location:")
    
    persona_key = st.sidebar.selectbox(
        "Historical Guide:",
        list(personas.keys()),
        format_func=lambda x: personas[x]['name']
    )
    
    landmark_key = st.sidebar.selectbox(
        "Landmark:",
        list(landmarks.keys()),
        format_func=lambda x: landmarks[x]['name']
    )
    
    # Load the AI generator
    with st.spinner("Lade AI-Modell..."):
        generator = load_simple_generator()
    
    # Main interface
    st.title("🎭 Vienna AI Historical Guide")
    st.markdown(f"**Your Guide:** {personas[persona_key]['name']}")
    st.markdown(f"**Location:** {landmarks[landmark_key]['name']}")
    
    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Display current persona and landmark info
    with st.expander("ℹ️ About Your Guide and Location"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Your Historical Guide")
            st.write(f"**Name:** {personas[persona_key]['name']}")
            st.write(f"**Traits:** {personas[persona_key]['traits']}")
            st.write(f"**Importance:** {personas[persona_key]['importance']}")
            st.write(f"**Tourist Appeal:** {personas[persona_key]['tourist_attraction']}")
        with col2:
            st.subheader("Location")
            st.write(f"**Name:** {landmarks[landmark_key]['name']}")
            st.write(f"**Description:** {landmarks[landmark_key]['description']}")
            st.write(f"**History:** {landmarks[landmark_key]['history']}")
    
    # Chat interface
    st.subheader("💬 Ask Your Question:")
    
    # User input
    user_input = st.text_input(
        "Question:",
        placeholder="e.g. 'Tell me about the history of this place'"
    )
    
    if user_input:
        # Add user message to history
        st.session_state.history.append(("user", user_input))
        
        # Generate response
        prompt = format_prompt(
            persona=personas[persona_key],
            landmark=landmarks[landmark_key],
            user_query=user_input
        )
        
        with st.spinner("Generating response..."):
            reply = generate_response(generator, prompt)
        
        # Add AI response to history
        st.session_state.history.append((persona_key, reply))
    
    # Display chat history
    st.subheader("🗨️ Conversation:")
    
    for i, (speaker, message) in enumerate(st.session_state.history):
        if speaker == "user":
            st.chat_message("user").write(f"**You:** {message}")
        else:
            persona_name = personas[speaker]['name']
            st.chat_message("assistant").write(f"**{persona_name}:** {message}")
    
    # Clear chat button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.history = []
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("*Vienna AI Historical Guide - Experience Vienna through the eyes of historical figures*")

if __name__ == "__main__":
    main()
