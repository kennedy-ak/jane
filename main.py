

# import streamlit as st
# import requests
# from PIL import Image
# import io
# import base64
# from transformers import pipeline, AutoProcessor, AutoModel
# import torch
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import pickle
# import os
# import anthropic
# from dotenv import load_dotenv

# load_dotenv()  # Load environment variables from .env file
# # Configure the page
# st.set_page_config(
#     page_title="Celebrity Detection AI",
#     page_icon="⭐",
#     layout="wide"
# )

# # Initialize the vision model with local caching
# @st.cache_resource
# def load_model():
#     try:
#         # Create local cache directory
#         cache_dir = "./model_cache"
#         os.makedirs(cache_dir, exist_ok=True)
        
#         # Using CLIP model for image-text matching with local cache
#         processor = AutoProcessor.from_pretrained(
#             "openai/clip-vit-base-patch32",
#             cache_dir=cache_dir
#         )
#         model = AutoModel.from_pretrained(
#             "openai/clip-vit-base-patch32",
#             cache_dir=cache_dir
#         )
#         return processor, model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None, None

# # Initialize Claude API
# @st.cache_resource
# def init_claude_client():
#     try:
#         # You need to set your Anthropic API key as an environment variable
#         # or in Streamlit secrets
#         api_key =  os.getenv("ANTHROPIC_API_KEY")   
#         if not api_key:
#             st.warning("⚠️ Claude API key not found. Celebrity info feature will be disabled.")
#             return None
#         return anthropic.Anthropic(api_key=api_key)
#     except Exception as e:
#         st.warning(f"Could not initialize Claude API: {e}")
#         return None

# def get_celebrity_info(celebrity_name, claude_client):
#     """Get brief information about the celebrity using Claude API"""
#     if not claude_client:
#         return "Celebrity information feature unavailable (API key not configured)."
    
#     try:
#         prompt = f"""Provide a brief 2-3 sentence summary about {celebrity_name}. 
#         Include their profession, notable achievements, and current status. 
#         Keep it concise and factual."""
        
#         message = claude_client.messages.create(
#             model="claude-3-sonnet-20240229",
#             max_tokens=150,
#             messages=[{"role": "user", "content": prompt}]
#         )
        
#         return message.content[0].text
#     except Exception as e:
#         return f"Unable to fetch information about {celebrity_name}."

# # Define our celebrities
# FOOTBALLERS = [
#     "Lionel Messi",
#     "Cristiano Ronaldo", 
#     "Neymar Jr",
#     "Kylian Mbappé",
#     "Erling Haaland"
# ]

# GHANAIAN_MUSICIANS = [
#     "Sarkodie",
#     "Stonebwoy",
#     "Shatta Wale", 
#     "Black Sherif",
#     "King Promise"
# ]

# ALL_CELEBRITIES = FOOTBALLERS + GHANAIAN_MUSICIANS

# def predict_celebrity(image, processor, model):
#     """
#     Predict which celebrity the image most likely contains
#     """
#     try:
#         # Create text descriptions for each celebrity
#         text_descriptions = [f"a photo of {celebrity}" for celebrity in ALL_CELEBRITIES]
        
#         # Process the image and text
#         inputs = processor(
#             text=text_descriptions, 
#             images=image, 
#             return_tensors="pt", 
#             padding=True
#         )
        
#         # Get predictions
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits_per_image = outputs.logits_per_image
#             probs = logits_per_image.softmax(dim=1)
        
#         # Get the celebrity with highest probability
#         predicted_idx = torch.argmax(probs, dim=1).item()
#         confidence = probs[0][predicted_idx].item()
        
#         predicted_celebrity = ALL_CELEBRITIES[predicted_idx]
        
#         return predicted_celebrity, confidence, probs[0].tolist()
        
#     except Exception as e:
#         st.error(f"Error in prediction: {e}")
#         return None, 0, []

# def main():
#     # Header
#     st.title("🌟 Celebrity Detection AI")
#     st.markdown("**Detect Popular Footballers & Ghanaian Musicians**")
    
#     # Sidebar with information
#     with st.sidebar:

   
            
#         st.markdown("---")
#         st.markdown("**How it works:**")
#         st.markdown("1. Upload an image")
#         st.markdown("2. AI analyzes the image")
#         st.markdown("3. Get prediction with confidence score")
    
#     # Main content area
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.header("📸 Upload Image")
        
#         # File uploader
#         uploaded_file = st.file_uploader(
#             "Choose an image file",
#             type=['jpg', 'jpeg', 'png', 'webp'],
#             help="Upload a clear image of one of the supported celebrities"
#         )
        
#         # Sample images section
#         st.subheader("🎯 Or try sample detection")
#         if st.button("🏃‍♂️ Test with Sample Image"):
#             # Create a placeholder image for demo
#             sample_image = Image.new('RGB', (300, 300), color='lightblue')
#             uploaded_file = sample_image
    
#     with col2:
#         st.header("🤖 AI Detection Results")
        
#         if uploaded_file is not None:
#             # Load the model
#             processor, model = load_model()
            
#             if processor is not None and model is not None:
#                 # Display the uploaded image
#                 if isinstance(uploaded_file, Image.Image):
#                     image = uploaded_file
#                 else:
#                     image = Image.open(uploaded_file)
                
#                 st.image(image, caption="Uploaded Image", use_container_width=True)
                
#                 # Make prediction
#                 with st.spinner("🔍 Analyzing image..."):
#                     predicted_celebrity, confidence, all_probs = predict_celebrity(
#                         image, processor, model
#                     )
                
#                 if predicted_celebrity:
#                     # Display results
#                     st.success(f"**Prediction: {predicted_celebrity}**")
#                     st.metric("Confidence", f"{confidence:.2%}")
                    
#                     # Determine category
#                     if predicted_celebrity in FOOTBALLERS:
#                         st.info("⚽ Category: Footballer")
#                     else:
#                         st.info("🎵 Category: Ghanaian Musician")
                    
#                     # Show confidence bar
#                     st.progress(confidence)
                    
#                     # Get celebrity information using Claude API
#                     claude_client = init_claude_client()
#                     if confidence > 0.3:  # Only fetch info if confidence is reasonable
#                         with st.spinner("📖 Fetching celebrity information..."):
#                             celebrity_info = get_celebrity_info(predicted_celebrity, claude_client)
                        
#                         st.subheader("ℹ️ About This Celebrity")
#                         st.info(celebrity_info)

                    
#                     # Confidence interpretation
#                     if confidence > 0.7:
#                         st.success("🎯 High confidence prediction!")
#                     elif confidence > 0.4:
#                         st.warning("🤔 Moderate confidence - consider uploading a clearer image")
#                     else:
#                         st.error("❓ Low confidence - the person might not be in our database or image quality is poor")
#             else:
#                 st.error("❌ Failed to load the AI model. Please refresh the page.")
#         else:
#             st.info("👆 Please upload an image to start detection")
            
#             # Instructions
#             st.markdown("### 💡 Tips for better results:")
#             st.markdown("- Use clear, high-quality images")
#             st.markdown("- Ensure the person's face is clearly visible")
#             st.markdown("- Avoid group photos or multiple people")
#             st.markdown("- Good lighting improves accuracy")

#     # Footer
#     st.markdown("---")
#     st.markdown("### ⚡ About this AI")
#     st.markdown("""
#     This application uses advanced computer vision and natural language processing 
#     to identify celebrities from images. It combines:
#     - **CLIP Model**: For image-text understanding
#     - **Claude API**: For celebrity information retrieval
#     - **Streamlit**: For the web interface
#     - **Machine Learning**: For accurate predictions

#     **Note**: This is a demonstration app. Accuracy may vary based on image quality 
#     and similarity to training data. Model is cached locally to avoid re-downloading.
#     """)

#     # Configuration section
#     with st.expander("⚙️ Configuration"):
#         st.markdown("**API Configuration:**")
#         api_key_status = "✅ Configured" if (st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")) else "❌ Not configured"
#         st.write(f"Claude API Key: {api_key_status}")
        
#         st.markdown("**Model Cache:**")
#         cache_exists = "✅ Cached" if os.path.exists("./model_cache") else "❌ Not cached"
#         st.write(f"CLIP Model: {cache_exists}")
        
#         if st.button("🗑️ Clear Model Cache"):
#             try:
#                 import shutil
#                 if os.path.exists("./model_cache"):
#                     shutil.rmtree("./model_cache")
#                     st.success("Model cache cleared! Restart the app to re-download.")
#                 else:
#                     st.info("No cache to clear.")
#             except Exception as e:
#                 st.error(f"Error clearing cache: {e}")

# if __name__ == "__main__":
#     main()



import streamlit as st
import requests
from PIL import Image
import io
import base64
from transformers import pipeline, AutoProcessor, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import openai
from dotenv import load_dotenv
import pyttsx3
import threading

load_dotenv()  # Load environment variables from .env file

# Configure the page
st.set_page_config(
    page_title="Celebrity Detection AI",
    page_icon="⭐",
    layout="wide"
)

# Initialize the vision model with local caching
@st.cache_resource
def load_model():
    try:
        # Create local cache directory
        cache_dir = "./model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Using CLIP model for image-text matching with local cache
        processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir
        )
        model = AutoModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir
        )
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Initialize text-to-speech engine
@st.cache_resource
def init_tts_engine():
    try:
        engine = pyttsx3.init()
        # Set properties for better speech
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        
        # Try to set a more natural voice (optional)
        voices = engine.getProperty('voices')
        if voices:
            # Use first available voice, or you can filter for specific voice types
            engine.setProperty('voice', voices[0].id)
        
        return engine
    except Exception as e:
        st.warning(f"Text-to-speech not available: {e}")
        return None

def speak_text(text, engine):
    """Speak the given text using text-to-speech"""
    if engine is None:
        st.error("Text-to-speech engine not available")
        return
    
    def run_tts():
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            st.error(f"Error in text-to-speech: {e}")
    
    # Run TTS in a separate thread to avoid blocking the UI
    tts_thread = threading.Thread(target=run_tts)
    tts_thread.daemon = True
    tts_thread.start()
@st.cache_resource
def init_openai_client():
    try:
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.warning("⚠️ OpenAI API key not found. Celebrity info feature will be disabled.")
            return None
        
        # Set the API key for openai
        openai.api_key = api_key
        return True
    except Exception as e:
        st.warning(f"Could not initialize OpenAI API: {e}")
        return None

def get_celebrity_info(celebrity_name, openai_available):
    """Get brief information about the celebrity using OpenAI API"""
    if not openai_available:
        return "Celebrity information feature unavailable (API key not configured)."
    
    try:
        prompt = f"""Provide a brief 2-3 sentence summary about {celebrity_name}. 
        Include their profession, notable achievements, and current status. 
        Keep it concise and factual."""
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Unable to fetch information about {celebrity_name}. Error: {str(e)}"

# Define our celebrities
FOOTBALLERS = [
    "Lionel Messi",
    "Cristiano Ronaldo", 
    "Neymar Jr",
    "Kylian Mbappé",
    "Erling Haaland"
]

GHANAIAN_MUSICIANS = [
    "Sarkodie",
    "Stonebwoy",
    "Shatta Wale", 
    "Black Sherif",
    "King Promise"
]

GHANAIAN_FEMALE_CELEBRITIES = [
    "Wendy Shay",      # Musician
    "Efya",            # Musician
    "MzVee",           # Musician
    "Jackie Appiah",   # Actress
    "Nadia Buari"      # Actress
]

ALL_CELEBRITIES = FOOTBALLERS + GHANAIAN_MUSICIANS + GHANAIAN_FEMALE_CELEBRITIES 

def predict_celebrity(image, processor, model):
    """
    Predict which celebrity the image most likely contains
    """
    try:
        # Create text descriptions for each celebrity
        text_descriptions = [f"a photo of {celebrity}" for celebrity in ALL_CELEBRITIES]
        
        # Process the image and text
        inputs = processor(
            text=text_descriptions, 
            images=image, 
            return_tensors="pt", 
            padding=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get the celebrity with highest probability
        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_idx].item()
        
        predicted_celebrity = ALL_CELEBRITIES[predicted_idx]
        
        return predicted_celebrity, confidence, probs[0].tolist()
        
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, 0, []

def main():
    # Header
    st.title("🌟 Celebrity Detection AI")
    st.markdown("**Detect Popular Footballers & Ghanaian Musicians**")
    
    # Sidebar with information
    with st.sidebar:
        st.subheader("⚽ Footballers")
        for footballer in FOOTBALLERS:
            st.write(f"• {footballer}")
            
        st.subheader("🎵 Ghanaian Musicians")  
        for musician in GHANAIAN_MUSICIANS:
            st.write(f"• {musician}")
            
        st.subheader("🌟 Ghanaian Female Celebrities")
        for celebrity in GHANAIAN_FEMALE_CELEBRITIES:
            st.write(f"• {celebrity}")
            
        st.markdown("---")
        st.markdown("**How it works:**")
        st.markdown("1. Upload an image")
        st.markdown("2. AI analyzes the image")
        st.markdown("3. Get prediction with confidence score")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload a clear image of one of the supported celebrities"
        )
        
        # Sample images section
        st.subheader("🎯 Or try sample detection")
        if st.button("🏃‍♂️ Test with Sample Image"):
            # Create a placeholder image for demo
            sample_image = Image.new('RGB', (300, 300), color='lightblue')
            uploaded_file = sample_image
    
    with col2:
        st.header("🤖 AI Detection Results")
        
        if uploaded_file is not None:
            # Load the model
            processor, model = load_model()
            
            if processor is not None and model is not None:
                # Display the uploaded image
                if isinstance(uploaded_file, Image.Image):
                    image = uploaded_file
                else:
                    image = Image.open(uploaded_file)
                
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Make prediction
                with st.spinner("🔍 Analyzing image..."):
                    predicted_celebrity, confidence, all_probs = predict_celebrity(
                        image, processor, model
                    )
                
                if predicted_celebrity:
                    # Display results
                    st.success(f"**Prediction: {predicted_celebrity}**")
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Determine category
                    if predicted_celebrity in FOOTBALLERS:
                        st.info("⚽ Category: Footballer")
                    elif predicted_celebrity in GHANAIAN_MUSICIANS:
                        st.info("🎵 Category: Ghanaian Musician")
                    else:
                        st.info("🌟 Category: Ghanaian Female Celebrity")
                    
                    # Show confidence bar
                    st.progress(confidence)
                    
                    # Get celebrity information using OpenAI API
                    openai_available = init_openai_client()
                    if confidence > 0.3:  # Only fetch info if confidence is reasonable
                        with st.spinner("📖 Fetching celebrity information..."):
                            celebrity_info = get_celebrity_info(predicted_celebrity, openai_available)
                        
                        st.subheader("ℹ️ About This Celebrity")
                        st.info(celebrity_info)
                        
                        # Add text-to-speech button
                        col_read1, col_read2 = st.columns([1, 3])
                        with col_read1:
                            if st.button("🔊 Read Aloud", key="read_celebrity_info"):
                                tts_engine = init_tts_engine()
                                if tts_engine:
                                    # Create a comprehensive text to read
                                    text_to_read = f"Celebrity detected: {predicted_celebrity}. {celebrity_info}"
                                    with st.spinner("🗣️ Reading information..."):
                                        speak_text(text_to_read, tts_engine)
                                    st.success("✅ Reading complete!")
                                else:
                                    st.error("Text-to-speech not available on this system")
                        
                        with col_read2:
                            st.caption("Click the button above to hear the celebrity information")


                    
                    # Confidence interpretation
                    if confidence > 0.7:
                        st.success("🎯 High confidence prediction!")
                    elif confidence > 0.4:
                        st.warning("🤔 Moderate confidence - consider uploading a clearer image")
                    else:
                        st.error("❓ Low confidence - the person might not be in our database or image quality is poor")
            else:
                st.error("❌ Failed to load the AI model. Please refresh the page.")
        else:
            st.info("👆 Please upload an image to start detection")
            
            # Instructions
            st.markdown("### 💡 Tips for better results:")
            st.markdown("- Use clear, high-quality images")
            st.markdown("- Ensure the person's face is clearly visible")
            st.markdown("- Avoid group photos or multiple people")
            st.markdown("- Good lighting improves accuracy")

    # Footer
    st.markdown("---")
    st.markdown("### ⚡ About this AI")
    st.markdown("""
    This application uses advanced computer vision and natural language processing 
    to identify celebrities from images. It combines:
    - **CLIP Model**: For image-text understanding
    - **OpenAI API**: For celebrity information retrieval
    - **Streamlit**: For the web interface
    - **Machine Learning**: For accurate predictions
    
    **Setup Instructions:**
    1. Install dependencies: `pip install streamlit transformers torch pillow scikit-learn numpy openai python-dotenv pyttsx3`
    2. Create a `.env` file and add your OpenAI API key: `OPENAI_API_KEY=your-api-key-here`
    3. Run: `streamlit run celebrity_detector.py`
    
    **Note**: This is a demonstration app. Accuracy may vary based on image quality 
    and similarity to training data. Model is cached locally to avoid re-downloading.
    """)

    # Configuration section
    with st.expander("⚙️ Configuration"):
        st.markdown("**Text-to-Speech:**")
        tts_status = "✅ Available" if init_tts_engine() else "❌ Not available"
        st.write(f"TTS Engine: {tts_status}")
        
        st.markdown("**API Configuration:**")
        api_key_status = "✅ Configured" if os.getenv("OPENAI_API_KEY") else "❌ Not configured"
        st.write(f"OpenAI API Key: {api_key_status}")
        
        st.markdown("**Model Cache:**")
        cache_exists = "✅ Cached" if os.path.exists("./model_cache") else "❌ Not cached"
        st.write(f"CLIP Model: {cache_exists}")
        
        if st.button("🗑️ Clear Model Cache"):
            try:
                import shutil
                if os.path.exists("./model_cache"):
                    shutil.rmtree("./model_cache")
                    st.success("Model cache cleared! Restart the app to re-download.")
                else:
                    st.info("No cache to clear.")
            except Exception as e:
                st.error(f"Error clearing cache: {e}")

if __name__ == "__main__":
    main()
