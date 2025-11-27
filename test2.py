import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient
import io
import os
import tempfile

# --- 1. CONFIGURATION ---
VISION_MODEL = "facebook/detr-resnet-50"
LLAMA_MODEL = "openai/gpt-oss-120b"

st.set_page_config(page_title="Recycling Advisor", layout="wide")

# --- 2. SETUP ---
hf_token = "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"
if not hf_token:
    st.error("❌ HF_TOKEN is missing.")
    st.stop()

client = InferenceClient(token=hf_token)


# --- 3. FUNCTIONS ---
def clean_response(text):
    if not text:
        return text

    # Remove html breaks
    text = (
        text.replace("<br>", "\n")
            .replace("<br/>", "\n")
            .replace("<br />", "\n")
    )

    # Remove zero-width characters and formatting junk
    bad_chars = [
        "\u200b",  # zero-width space
        "\ufeff",  # BOM
        "\u200e",  # LTR mark
        "\u200f",  # RTL mark
    ]
    for ch in bad_chars:
        text = text.replace(ch, "")

    # Normalize all weird hyphens to normal hyphen
    hyphens = ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014"]
    for hy in hyphens:
        text = text.replace(hy, "-")

    return text


def detect_objects(image):
    """
    Detects objects using DETR. Uses tempfile to ensure the client gets a named file path,
    and includes manual cleanup to resolve the Permission Denied error.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name

        image.save(temp_path, format='JPEG')

        results = client.object_detection(image=temp_path, model=VISION_MODEL)

        detected_items = []
        if isinstance(results, list):
            for item in results:
                if item.get('score', 0) > 0.7:
                    detected_items.append(item.get('label', 'unknown'))
        else:
            return f"Error: Unexpected response format from API: {results}"

        unique_items = list(set(detected_items))

        if not unique_items:
            return "Unidentified trash item"

        return ", ".join(unique_items)

    except Exception as e:
        if 'Permission denied' in str(e) and temp_path:
            return f"Error: Permission denied for {temp_path}. Check admin privileges."
        return f"Error: {str(e)}"
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def get_advice_llama(detected_list):
    """
    Asks Llama 3.2 how to recycle the detected items.
    """
    system_prompt = (
        '''You are a waste segregation and recycling process expert. I will provide a list of items detected in a trash pile.
        For EACH item in the list, tell me:
        1. The Material (Plastic, Glass, Metal, etc.).
        2. Preparation (Rinse, Crush, Separate components, Empty contents etc.).
        3. The Correct Bin / Collection Method (Curbside, Drop-off Center, Hazardous Waste, Compost etc.).
        4. The Post-Collection Recycling / Re-use Action (example: melted down, shredded, reused, repurposed).
        Format as a clean table or list.'''
    )

    user_prompt = f"Detected items: {detected_list}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat_completion(
            model=LLAMA_MODEL,
            messages=messages,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Llama API Error: {e}"


# --- 4. ENHANCED UI ---

st.markdown("""
    <h1 style="text-align:center; font-size:40px;">♻️ Recycling Advisor</h1>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center; color:grey; font-size:16px; margin-top:-10px;">
Vision Model: <b>{VISION_MODEL}</b> &nbsp; | &nbsp; Reasoning Model: <b>{LLAMA_MODEL}</b>
</div>
""", unsafe_allow_html=True)

st.write("")

with st.container():
    st.markdown("""
    <div style="padding:20px; border-radius:15px; background: #1f1f1f55;">
        <h3 style="margin-bottom:5px;">📤 Upload Trash Image</h3>
        <p style="opacity:0.8; margin-top:-8px;">Supported formats: JPG • PNG</p>
    </div>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

st.write("")

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")
        st.write("")

        with st.spinner("🔍 Scanning objects in image..."):
            detected_text = detect_objects(image)

        st.write("")
        st.markdown("### 🧩 Detection Results")

        if "Error" in detected_text:
            st.error(f"⚠️ Vision Model Error: {detected_text}")

        else:
            if detected_text == "Unidentified trash item":
                st.warning("⚠️ Low confidence detection. Please identify manually.")
                detected_text = st.text_input(
                    "Type what the object appears to be:",
                    "General mixed waste"
                )
            else:
                st.success(f"**Detected items:** {detected_text}")

            st.write("")

            st.markdown("### ♻️ Recycling Instructions")

            advice_box = st.container()

            with st.spinner("🧠 Preparing recycling advice..."):
                advice = get_advice_llama(detected_text)

            cleaned_advice = clean_response(advice)

            with advice_box:
                # Markdown now renders correctly
                st.markdown(cleaned_advice)


    except Exception as e:
        st.error(f"Processing Error: {e}")
