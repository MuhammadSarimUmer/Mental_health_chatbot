import os, json, re, datetime, tempfile
from pathlib import Path
from gtts import gTTS
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import gradio as gr

# API KEY - reads from HuggingFace Secret
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY secret not set. Add it in Space Settings > Secrets.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 1. KNOWLEDGE BASE
MENTAL_HEALTH_DOCS = [
    """Anxiety is a natural response to stress. Common symptoms include racing heart, sweating, trembling, shortness of breath, and persistent worry.
    Cognitive Behavioral Therapy (CBT) is highly effective for anxiety. It helps identify and challenge distorted thought patterns.
    Grounding techniques like the 5-4-3-2-1 method can interrupt anxiety spirals: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.
    Regular exercise, especially aerobic activity for 30 minutes 3-5 times a week, significantly reduces anxiety symptoms.
    Limiting caffeine and alcohol can reduce anxiety. Both substances can trigger or worsen symptoms.""",

    """Depression involves persistent sadness, loss of interest, changes in sleep and appetite, fatigue, and feelings of worthlessness.
    Behavioral activation means scheduling small enjoyable activities and is a core depression treatment.
    Sleep hygiene is critical. Consistent sleep and wake times and limiting screens before bed improve mood significantly.
    Social connection is protective. Isolation worsens depression. Even brief social contact helps.
    Self-compassion involves treating yourself with the same kindness you would offer a good friend.""",

    """Breathing exercises activate the parasympathetic nervous system and reduce stress hormones within minutes.
    Box breathing: inhale for 4 counts, hold for 4, exhale for 4, hold for 4. Repeat 4-6 times.
    4-7-8 breathing: inhale for 4 counts, hold for 7, exhale slowly for 8. This promotes relaxation and can aid sleep.
    Diaphragmatic breathing: breathe into your belly, not chest. Only the belly hand should rise.
    Alternate nostril breathing from yoga reduces stress and improves focus.""",

    """Mindfulness is the practice of non-judgmental present-moment awareness. It reduces rumination, stress, and emotional reactivity.
    Body scan meditation: slowly move attention from feet to head, noticing sensations without judgment. Takes 5-20 minutes.
    MBSR (Mindfulness-Based Stress Reduction) is an 8-week program proven to reduce anxiety, depression, and chronic pain.
    Mindful walking: walk slowly, feel each step, notice your breath and surroundings.
    Journaling 3 things you are grateful for daily rewires the brain toward positivity and reduces depression over time.""",

    """Crisis resources: If you are in immediate danger, call emergency services (911 in USA, 115 in Pakistan, 999 in UK).
    Pakistan crisis line: Umang helpline 0317-4288665, available Monday-Saturday 3pm-9pm.
    iCall India: 9152987821. Vandrevala Foundation 24/7: 1860-2662-345.
    Samaritans UK: 116 123 (free, 24/7). Crisis Text Line US: Text HOME to 741741.
    Reaching out for help is a sign of strength, not weakness. You are not alone.""",

    """Panic attacks are intense surges of fear that peak within minutes. They are not dangerous but feel terrifying.
    During a panic attack: remind yourself this will pass, I am safe, this is temporary.
    Cold water on the face triggers the diving reflex and rapidly slows heart rate.
    Progressive muscle relaxation: systematically tense and release muscle groups from toes to face.
    Instead of fighting panic, accepting it reduces its power.""",

    """Stress management involves identifying stressors, building coping resources, and creating recovery time.
    Time management techniques like the Pomodoro method (25 min focus, 5 min break) reduce overwhelm.
    Setting boundaries at work and in relationships protects mental energy. Saying no is self-care.
    Nature exposure (even 20 minutes) reduces cortisol.
    Creative expression such as art, music, and writing provides emotional release and reduces stress hormones.""",

    """Self-care is not selfish. It includes physical, emotional, social, spiritual, and professional dimensions.
    Physical self-care: regular sleep, nutrition, movement, medical checkups, limiting substances.
    Emotional self-care: therapy, journaling, processing emotions, setting limits on news and social media.
    Social self-care: nurturing supportive relationships, spending time with loved ones, seeking community.
    Professional help: therapists, counselors, and psychiatrists provide evidence-based treatment. There is no shame in seeking help."""
]

# 2. BUILD RAG
print("Building RAG knowledge base...")
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
docs = []
for text in MENTAL_HEALTH_DOCS:
    chunks = splitter.split_text(text)
    docs.extend([Document(page_content=c) for c in chunks])

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("RAG ready!")

# 3. GROQ CLIENT
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are Echo, a compassionate, warm, and knowledgeable mental health companion. "
    "You are NOT a replacement for professional therapy, but you provide genuine emotional support, "
    "psychoeducation, and coping strategies.\n\n"
    "Your personality:\n"
    "- Warm, empathetic, non-judgmental, and gently encouraging\n"
    "- You validate feelings before offering advice\n"
    "- You use simple, accessible language\n"
    "- You celebrate small wins enthusiastically\n"
    "- You gently encourage professional help when appropriate\n"
    "- You never dismiss or minimize someone's pain\n\n"
    "When a crisis is detected, always provide:\n"
    "1. Immediate validation and calm presence\n"
    "2. A grounding technique right away\n"
    "3. Crisis resources (Pakistan: Umang 0317-4288665, International: iasp.info)\n"
    "4. Encouragement to reach out to someone they trust\n\n"
    "Keep responses concise (2-4 paragraphs) unless explaining a technique."
)

# 4. CRISIS DETECTION
CRISIS_KEYWORDS = [
    r"\bsuicid", r"\bkill myself\b", r"\bend my life\b", r"\bwant to die\b",
    r"\bno reason to live\b", r"\bhurt myself\b", r"\bself.harm",
    r"\bcant go on\b", r"\bgive up on life\b", r"\bnot worth living\b",
    r"\bwish i was dead\b", r"\bdie by suicide\b"
]

def detect_crisis(text):
    return any(re.search(kw, text.lower()) for kw in CRISIS_KEYWORDS)

# 5. CHAT STORAGE - use /tmp for HuggingFace
CHAT_FILE = "/tmp/echo_chat_history.json"

# 5. CHAT STORAGE - Disabled global file to ensure user privacy

def load_history():
    # Return an empty list so every new visitor gets a fresh, private screen
    return []

def save_history(history):
    # Pass does nothing. We let Gradio's gr.State handle the memory privately!
    pass

def delete_history():
    # Just clears the screen without looking for a file
    return []

# 6. TTS
def text_to_speech(text, lang="en"):
    try:
        clean = re.sub(r'[*_#`]', '', text)
        clean = re.sub(r'\s+', ' ', clean).strip()[:800]
        clean = clean.encode('ascii', 'ignore').decode('ascii')
        tts = gTTS(text=clean, lang=lang, slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir="/tmp")
        tts.save(tmp.name)
        return tmp.name
    except Exception as e:
        print(f"TTS error: {e}")
        return None

# 7. RAG RETRIEVAL
def retrieve_context(query):
    results = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in results])

# 8. MAIN CHAT
MAX_CONTEXT_TURNS = 8

def clean_for_api(text):
    return text.encode('ascii', 'ignore').decode('ascii')

def chat_with_echo(user_message, conversation_history, enable_rag=True):
    is_crisis = detect_crisis(user_message)
    messages = [{"role": "system", "content": clean_for_api(SYSTEM_PROMPT)}]

    if enable_rag:
        ctx = retrieve_context(user_message)
        if ctx:
            messages.append({"role": "system", "content": clean_for_api("Relevant knowledge (weave naturally):\n" + ctx)})

    if is_crisis:
        messages.append({"role": "system", "content": "IMPORTANT: This person may be in crisis. Respond with immediate warmth, a brief grounding exercise, crisis resources (Pakistan: Umang 0317-4288665), and encourage them to seek help now."})

    for turn in conversation_history[-(MAX_CONTEXT_TURNS * 2):]:
        messages.append({"role": turn["role"], "content": clean_for_api(turn["content"])})

    messages.append({"role": "user", "content": clean_for_api(user_message)})

    response = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=600, temperature=0.75
    )
    reply = response.choices[0].message.content

    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": reply})
    save_history(conversation_history)

    return reply, text_to_speech(reply), conversation_history, is_crisis

# 9. MOOD RESPONSES
MOOD_PROMPTS = {
    "😊 Happy": "The user is feeling happy. Respond with genuine excitement, celebrate with them, ask what's making them happy, and encourage them to savour this feeling.",
    "😢 Sad": "The user is feeling sad. Respond with deep empathy, validate that sadness is okay, gently ask what's going on, and offer one small comforting suggestion.",
    "😰 Anxious": "The user is feeling anxious. Respond calmly, validate their anxiety, immediately offer box breathing (inhale 4, hold 4, exhale 4), and remind them anxiety passes.",
    "😤 Angry": "The user is feeling angry. Validate their anger without judgment, acknowledge it is a valid emotion, help them identify what might be underneath it.",
    "😴 Tired": "The user is feeling tired. Show compassion, ask if it is physical or emotional tiredness, gently affirm that rest is productive, and suggest one simple self-care action.",
    "😕 Lost": "The user is feeling lost and confused about life. Respond with deep understanding, normalize feeling lost, remind them uncertainty is part of growth.",
    "🥰 Grateful": "The user is feeling grateful. Celebrate this with them genuinely, explore what they are grateful for, explain the science of gratitude briefly.",
    "😶 Numb": "The user is feeling numb and disconnected. Respond with care, normalize emotional numbness as a protective response, and suggest a gentle grounding technique."
}

def handle_mood(mood_label, conversation_history):
    prompt = MOOD_PROMPTS.get(mood_label, f"The user is feeling {mood_label}. Respond empathetically.")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": clean_for_api(SYSTEM_PROMPT)},
            {"role": "user", "content": clean_for_api(prompt)}
        ],
        max_tokens=300, temperature=0.8
    )
    reply = response.choices[0].message.content
    conversation_history.append({"role": "user", "content": f"[Mood: {mood_label}]"})
    conversation_history.append({"role": "assistant", "content": reply})
    save_history(conversation_history)
    return reply, text_to_speech(reply), conversation_history

# 10. COPING TOOLS
COPING_TOOLS = {
    "Box Breathing": {
        "text": "**Box Breathing (4-4-4-4)**\n\n1. Inhale through your nose for 4 counts\n2. Hold for 4 counts\n3. Exhale through your mouth for 4 counts\n4. Hold empty for 4 counts\n\nRepeat 4-6 times. You will feel calmer within 2 minutes.",
        "tts": "Breathe in for 4. 1, 2, 3, 4. Hold for 4. 1, 2, 3, 4. Breathe out for 4. 1, 2, 3, 4. Hold for 4. 1, 2, 3, 4. Repeat 4 to 6 times."
    },
    "5-4-3-2-1 Grounding": {
        "text": "**5-4-3-2-1 Grounding**\n\n- 5 things you can SEE\n- 4 things you can TOUCH\n- 3 things you can HEAR\n- 2 things you can SMELL\n- 1 thing you can TASTE\n\nThis interrupts the anxiety cycle by anchoring you to the present.",
        "tts": "Name 5 things you can see. Now 4 things you can touch. Listen for 3 sounds. Notice 2 things you can smell. And 1 thing you can taste. You are here. You are safe."
    },
    "Thought Reframing": {
        "text": "**Cognitive Reframing**\n\n1. Identify the negative thought\n2. Ask: Is this 100% true? What is the evidence?\n3. Ask: What would I tell a friend thinking this?\n4. Replace with a kinder, more realistic version\n\nExample: 'I always fail' becomes 'I struggled here, but I have also succeeded many times.'",
        "tts": "Identify the negative thought. Ask: is this completely true? What would you say to a friend? Now find a more balanced, kinder way to see the situation."
    },
    "Body Scan": {
        "text": "**Body Scan Meditation**\n\n1. Close your eyes and take 3 deep breaths\n2. Bring attention to your feet\n3. Slowly move up through calves, thighs, hips, belly\n4. Continue through chest, shoulders, arms, hands\n5. Finish at neck, face, top of head\n\nAt each area, breathe into tension and release it on the exhale.",
        "tts": "Close your eyes. Take three deep breaths. Bring attention to your feet. Breathe into any tension. Slowly move up through your legs, hips, belly, chest, shoulders, arms, and hands. Finally your neck and face. Release any tension you find."
    },
    "Gratitude Practice": {
        "text": "**Gratitude Practice**\n\nThink of 3 things you are grateful for right now. They can be tiny:\n- A warm drink\n- Someone who made you smile\n- Your body keeping you alive\n\nWrite them down if you can. Do this every morning for 21 days.",
        "tts": "Think of three things you are grateful for. They can be small. Maybe warmth, someone who made you smile, or simply that you are here. Let yourself feel the gratitude for each one."
    },
    "Affirmations": {
        "text": "**Positive Affirmations**\n\nSay these slowly, out loud if possible:\n\n- I am worthy of love and belonging.\n- This feeling is temporary and will pass.\n- I have survived hard days before and I can do it again.\n- I am doing the best I can with what I have.\n- My resilience defines me, not my struggles.\n\nChoose one that resonates. Repeat it 5 times.",
        "tts": "I am worthy of love and belonging. This feeling is temporary and will pass. I have survived hard days before. I am doing the best I can. My resilience defines me. Choose the one that speaks to you and say it five times."
    }
}

# 11. RESOURCES
RESOURCES_TEXT = """## Emergency and Mental Health Resources

### Pakistan
- **Umang Helpline:** 0317-4288665 (Mon-Sat, 3pm-9pm)
- **Rozan Counselling:** 051-2890505
- **Emergency:** 115 or 1122

### International
- **Crisis Text Line (US):** Text HOME to 741741
- **Samaritans (UK):** 116 123 (24/7, free)
- **Lifeline (Australia):** 13 11 14
- **iCall (India):** 9152987821
- **Directory:** iasp.info/resources/Crisis_Centres

### Online Support
- **7 Cups:** 7cups.com
- **BetterHelp:** betterhelp.com

Reaching out is a sign of strength. You deserve support."""

RESOURCES_TTS = "In Pakistan, call Umang at 0317-4288665, Monday through Saturday, 3pm to 9pm. For emergencies call 115. In the US, text HOME to 741741. In the UK, Samaritans is at 116 123, available 24 hours. Visit iasp.info for your local crisis centre. Reaching out for help is one of the bravest things you can do. You are not alone."

# UI
# UI
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-main: #212121;
    --bg-sidebar: #171717;
    --text-primary: #ECECEC;
    --text-secondary: #B4B4B4;
    --border-light: rgba(255, 255, 255, 0.1);
    --msg-user: #2F2F2F;
    --msg-bot: transparent;
    --input-bg: #2F2F2F;
    --accent: #FFFFFF;
    --accent-hover: #ECECEC;
    --radius-full: 9999px;
    --radius-xl: 24px;
    --radius-lg: 16px;
    --radius-md: 8px;
}

/* Base Body */
body, .gradio-container {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 100vh !important;
    overflow: hidden !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #424242; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #565656; }

/* Force Text Colors */
p, h1, h2, h3, h4, span, div, label, li, a { color: var(--text-primary) !important; }
.text-muted { color: var(--text-secondary) !important; }

/* Main Layout Grid */
.main-layout {
    display: flex !important;
    flex-direction: row !important;
    height: 100vh !important;
    width: 100vw !important;
    gap: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
}

/* -----------------------------
   SIDEBAR STYLING
----------------------------- */
.sidebar {
    background-color: var(--bg-sidebar) !important;
    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;
    height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 16px 12px !important;
    border-right: none !important;
}

.new-chat-btn {
    background: transparent !important;
    color: var(--text-primary) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 12px 14px !important;
    text-align: left !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    display: flex !important;
    justify-content: flex-start !important;
    transition: background 0.2s !important;
}
.new-chat-btn:hover { background: #202123 !important; }

.sidebar-tools {
    margin-top: 32px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
}

.tool-btn {
    background: transparent !important;
    border: none !important;
    color: var(--text-primary) !important;
    text-align: left !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    border-radius: var(--radius-md) !important;
}
.tool-btn:hover { background: #202123 !important; }

/* -----------------------------
   CHAT AREA STYLING
----------------------------- */
.chat-container {
    flex-grow: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100vh !important;
    position: relative !important;
    background-color: var(--bg-main) !important;
}

.chatbot-wrap {
    flex-grow: 1 !important;
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
    overflow-y: auto !important;
    margin-bottom: 90px !important; /* Space for input */
}

/* Chat Messages */
.message-wrap { gap: 0 !important; padding: 24px 0 !important; }

/* Center the chat content */
.message {
    max-width: 768px !important; /* Standard ChatGPT width */
    margin: 0 auto !important;
    padding: 12px 24px !important;
    width: 100% !important;
}

/* User Message: Grey rounded bubble on the right */
.message.user {
    background-color: var(--msg-user) !important;
    border: none !important;
    border-radius: var(--radius-xl) !important;
    width: fit-content !important;
    max-width: 70% !important;
    margin-left: auto !important;
    margin-right: 24px !important;
    padding: 12px 20px !important;
    font-size: 16px !important;
}

/* Bot Message: Transparent, full width text */
.message.bot {
    background-color: var(--msg-bot) !important;
    border: none !important;
    border-radius: 0 !important;
    margin-left: 0 !important;
    font-size: 16px !important;
}
.message.bot p { line-height: 1.6 !important; }

/* -----------------------------
   INPUT AREA STYLING
----------------------------- */
.input-wrapper {
    position: absolute !important;
    bottom: 24px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 100% !important;
    max-width: 768px !important;
    padding: 0 24px !important;
    background: transparent !important;
    border: none !important;
}

.input-box {
    background-color: var(--input-bg) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-xl) !important;
    padding: 4px 16px !important;
    display: flex !important;
    align-items: center !important;
    box-shadow: 0 0 15px rgba(0,0,0,0.1) !important;
}

textarea, input[type="text"] {
    background-color: transparent !important;
    border: none !important;
    color: var(--text-primary) !important;
    font-size: 16px !important;
    box-shadow: none !important;
    padding: 14px 0 !important;
    resize: none !important;
}
textarea:focus { border: none !important; box-shadow: none !important; outline: none !important; }

/* Send Button: White circle */
button.send-btn {
    background-color: var(--accent) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: var(--radius-full) !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    padding: 0 !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    font-weight: bold !important;
    cursor: pointer !important;
}
button.send-btn:hover { background-color: var(--accent-hover) !important; }
button.send-btn span { color: #000000 !important; display: none; } /* Hide text if we just want a shape */
button.send-btn::after { content: "↑"; font-size: 20px; color: black; }

/* Hide default Gradio elements */
.form { border: none !important; background: transparent !important; box-shadow: none !important; }
footer { display: none !important; }
"""

# ==========================================
# 5. GRADIO APP DEFINITION
# ==========================================
def format_chat_for_gradio(history):
    messages = []
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
            messages.append({"role": "user", "content": history[i]["content"]})
            messages.append({"role": "assistant", "content": history[i+1]["content"]})
            i += 2
        else:
            i += 1
    return messages

def on_send(message, history_state, enable_tts):
    if not message.strip():
        return history_state, format_chat_for_gradio(history_state), None, ""
    reply, audio_path, updated_history, is_crisis = chat_with_echo(message, history_state)
    crisis_msg = "**Crisis support activated:** Umang 0317-4288665" if is_crisis else ""
    return updated_history, format_chat_for_gradio(updated_history), (audio_path if enable_tts else None), crisis_msg

def clear_chat():
    return [], [], None, ""

# Use Base theme to block Gradio styling overrides
with gr.Blocks(theme=gr.themes.Base(), css=CUSTOM_CSS, title="Echo") as app:
    history_state = gr.State([])

    with gr.Row(elem_classes=["main-layout"]):
        
        # --- LEFT SIDEBAR ---
        with gr.Column(elem_classes=["sidebar"], scale=0):
            # ChatGPT "New Chat" styling
            new_chat_btn = gr.Button("Echo Wellness", elem_classes=["new-chat-btn"])
            
            gr.Markdown("<div style='margin-top:24px; font-size:12px; color:#B4B4B4; padding-left:14px; font-weight:600;'>TOOLS & SETTINGS</div>")
            
            tts_toggle = gr.Checkbox(label="Voice Output", value=False, elem_classes=["tool-btn"])
            clear_btn = gr.Button("Clear Conversation", elem_classes=["tool-btn"])
            
            gr.Markdown("<div style='margin-top:auto; font-size:12px; color:#B4B4B4; padding-left:14px;'>Echo UI v2.0</div>")

        # --- RIGHT MAIN CHAT AREA ---
        with gr.Column(elem_classes=["chat-container"], scale=1):
            
            crisis_banner = gr.Markdown("", visible=True)
            
            # The Chat Display
            chatbot = gr.Chatbot(
                label="", 
                elem_classes=["chatbot-wrap"], 
                type="messages", 
                show_label=False,
                height=800
            )
            
            audio_out = gr.Audio(label="", autoplay=True, visible=False) 

            # The Bottom Input Area
            with gr.Column(elem_classes=["input-wrapper"]):
                with gr.Row(elem_classes=["input-box"]):
                    msg_input = gr.Textbox(
                        placeholder="Message Echo...", 
                        show_label=False, 
                        container=False, 
                        lines=1,
                        scale=9
                    )
                    send_btn = gr.Button("", elem_classes=["send-btn"], scale=1)

    # Event Listeners
    send_btn.click(
        fn=on_send, 
        inputs=[msg_input, history_state, tts_toggle],
        outputs=[history_state, chatbot, audio_out, crisis_banner]
    ).then(lambda: "", outputs=[msg_input])
    
    msg_input.submit(
        fn=on_send, 
        inputs=[msg_input, history_state, tts_toggle],
        outputs=[history_state, chatbot, audio_out, crisis_banner]
    ).then(lambda: "", outputs=[msg_input])
    
    clear_btn.click(fn=clear_chat, outputs=[history_state, chatbot, audio_out, crisis_banner])

if __name__ == "__main__":
    app.launch()