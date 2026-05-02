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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

:root {
    --bg-color: #F8FAFC; 
    --card-bg: #FFFFFF;
    --text-main: #1E293B;
    --text-light: #64748B;
    --accent: #0F766E; /* Deep, professional teal */
    --accent-hover: #0D9488;
    --user-msg: #F1F5F9;
    --bot-msg: #F0FDFA;
    --bot-border: #CCFBF1;
    --border-color: #E2E8F0;
    --radius-lg: 20px;
    --radius-sm: 12px;
    --shadow-sm: 0 2px 10px rgba(15, 23, 42, 0.04);
    --shadow-hover: 0 10px 25px rgba(15, 23, 42, 0.08);
}

/* 1. Fluid Background Animation (Breathing Effect) */
@keyframes breathingBg {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* 2. Slide & Fade Entry Animation */
@keyframes slideUpFade {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Base Body Override */
body, .gradio-container {
    background: linear-gradient(-45deg, #F8FAFC, #F1F5F9, #F0FDFA, #F8FAFC) !important;
    background-size: 400% 400% !important;
    animation: breathingBg 18s ease infinite !important;
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
    color: var(--text-main) !important;
}

.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    padding: 20px !important;
    animation: slideUpFade 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
}

/* Hide Gradio default styling */
.contain { border: none !important; background: transparent !important; }
.tabs { border: none !important; }
.tab-nav { border-bottom: 1px solid var(--border-color) !important; margin-bottom: 24px !important; gap: 16px !important; }
button.selected { border-bottom: 2px solid var(--accent) !important; color: var(--accent) !important; font-weight: 600 !important; background: transparent !important; }

/* Cards and Wrappers */
.chatbot-wrap, .output-text, .tabs > div > div {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow 0.3s ease !important;
}
.chatbot-wrap:hover { box-shadow: var(--shadow-hover) !important; }

/* Chatbot Bubbles - Professional Therapy App Style */
.chatbot-wrap { padding: 20px !important; }
.message-wrap { gap: 20px !important; }
.message.user {
    background-color: var(--user-msg) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-main) !important;
    border-radius: 18px 18px 4px 18px !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}
.message.bot {
    background-color: var(--bot-msg) !important;
    border: 1px solid var(--bot-border) !important;
    color: var(--text-main) !important;
    border-radius: 18px 18px 18px 4px !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
}

/* Input Row */
textarea, input[type="text"] {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-sm) !important;
    color: var(--text-main) !important;
    padding: 16px 20px !important;
    font-size: 15px !important;
    transition: all 0.3s ease !important;
}
textarea:focus { 
    border-color: var(--accent) !important; 
    box-shadow: 0 0 0 2px rgba(15, 118, 110, 0.1) !important; 
}

/* Animated Buttons */
button { transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important; }
button.primary {
    background-color: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
}
button.primary:hover { 
    background-color: var(--accent-hover) !important; 
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 15px rgba(15, 118, 110, 0.2) !important;
}
button.primary:active { transform: translateY(0) !important; }

/* Mood Buttons */
button.sm {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-light) !important;
    border-radius: 24px !important;
    font-weight: 500 !important;
    box-shadow: var(--shadow-sm) !important;
}
button.sm:hover { 
    background-color: var(--bot-msg) !important; 
    color: var(--accent) !important; 
    border-color: var(--bot-border) !important; 
    transform: translateY(-2px) !important;
}

/* Typography Overrides */
h1, h2, h3, p { color: var(--text-main) !important; }
hr { border-color: var(--border-color) !important; opacity: 0.5; }
"""

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
    crisis_msg = ""
    if is_crisis:
        crisis_msg = "Crisis support activated. Please reach out: Pakistan: Umang 0317-4288665 | US: Text HOME to 741741"
    return updated_history, format_chat_for_gradio(updated_history), (audio_path if enable_tts else None), crisis_msg

def on_mood(mood, history_state, enable_tts):
    reply, audio_path, updated_history = handle_mood(mood, history_state)
    return updated_history, format_chat_for_gradio(updated_history), (audio_path if enable_tts else None), reply

def on_coping_tool(tool_name, enable_tts):
    tool = COPING_TOOLS.get(tool_name)
    if not tool:
        return "Tool not found.", None
    return tool["text"], (text_to_speech(tool["tts"]) if enable_tts else None)

def on_resources_tts(enable_tts):
    return text_to_speech(RESOURCES_TTS) if enable_tts else None

def on_delete_chat():
    return delete_history(), [], None, "Chat history cleared."

def on_load():
    history = load_history()
    return history, format_chat_for_gradio(history)

with gr.Blocks(theme=gr.themes.Base(), css=CUSTOM_CSS, title="Echo | Wellness Companion") as app:
    history_state = gr.State([])

    # Redesigned Header: Professional Clinic aesthetic
    gr.HTML("""
    <div style="text-align:center;padding:48px 0 32px">
        <div style="font-size:3rem; margin-bottom:12px; transform: scale(1); transition: transform 0.3s ease;" onmouseover="this.style.transform='scale(1.1)'" onmouseout="this.style.transform='scale(1)'">🌱</div>
        <div style="font-size:2.4rem;font-weight:600;color:#0F766E;letter-spacing:-0.5px;font-family:'Plus Jakarta Sans', sans-serif;">Echo Wellness</div>
        <div style="color:#64748B;font-size:1.05rem;margin-top:8px;font-weight:400;">A private, supportive space for your mental wellbeing.</div>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("💬 Support Session"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=7):
                    chatbot = gr.Chatbot(label="", height=520, elem_classes=["chatbot-wrap"], type="messages", show_label=False)
                    crisis_banner = gr.Markdown("")
                    
                    with gr.Row():
                        msg_input = gr.Textbox(placeholder="I'm here to listen. Share what's on your mind...", show_label=False, container=False, lines=1, scale=5)
                        send_btn = gr.Button("Send", scale=1, variant="primary")
                    
                    with gr.Row():
                        tts_toggle = gr.Checkbox(label="Enable audio responses", value=False)
                    audio_out = gr.Audio(label="Audio Output", autoplay=True, visible=False) 

                with gr.Column(scale=3):
                    gr.Markdown("<div style='color:#1E293B; font-weight:600; margin-bottom:12px; font-size:1.1rem;'>How are you feeling today?</div>")
                    mood_output = gr.Markdown("", elem_classes=["output-text"])
                    moods = list(MOOD_PROMPTS.keys())
                    for i in range(0, len(moods), 2):
                        with gr.Row():
                            b1 = gr.Button(moods[i], size="sm")
                            b1.click(fn=on_mood, inputs=[gr.State(moods[i]), history_state, tts_toggle],
                                     outputs=[history_state, chatbot, audio_out, mood_output])
                            if i+1 < len(moods):
                                b2 = gr.Button(moods[i+1], size="sm")
                                b2.click(fn=on_mood, inputs=[gr.State(moods[i+1]), history_state, tts_toggle],
                                         outputs=[history_state, chatbot, audio_out, mood_output])
                    
                    gr.Markdown("<br>")
                    delete_btn = gr.Button("End Session", variant="stop", size="sm")

            send_btn.click(fn=on_send, inputs=[msg_input, history_state, tts_toggle],
                           outputs=[history_state, chatbot, audio_out, crisis_banner]).then(lambda: "", outputs=[msg_input])
            msg_input.submit(fn=on_send, inputs=[msg_input, history_state, tts_toggle],
                             outputs=[history_state, chatbot, audio_out, crisis_banner]).then(lambda: "", outputs=[msg_input])
            delete_btn.click(fn=on_delete_chat, outputs=[history_state, chatbot, audio_out, crisis_banner])

        with gr.Tab("🧘‍♀️ Coping Tools"):
            coping_tts_toggle = gr.Checkbox(label="Enable guided audio", value=False)
            with gr.Row():
                with gr.Column(scale=3):
                    coping_output = gr.Markdown("**Select a practice below to begin your guided exercise.**", elem_classes=["output-text"])
                with gr.Column(scale=2):
                    coping_audio = gr.Audio(label="Guided Audio", autoplay=True)
            gr.Markdown("<br>")
            with gr.Row():
                for tool_name in COPING_TOOLS.keys():
                    btn = gr.Button(tool_name, size="sm")
                    btn.click(fn=on_coping_tool, inputs=[gr.State(tool_name), coping_tts_toggle],
                              outputs=[coping_output, coping_audio])

        with gr.Tab("📞 Crisis Resources"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown(RESOURCES_TEXT, elem_classes=["output-text"])
                with gr.Column(scale=2):
                    res_tts_toggle = gr.Checkbox(label="Enable audio reading", value=False)
                    read_btn = gr.Button("Read Aloud", variant="primary")
                    resources_audio = gr.Audio(label="Audio Output", autoplay=True)
            read_btn.click(fn=on_resources_tts, inputs=[res_tts_toggle], outputs=[resources_audio])

        with gr.Tab("ℹ️ About Echo"):
            gr.Markdown("""
            ### Welcome to Echo Wellness
            Echo is designed to provide a safe, comforting space to process your thoughts and guide you through evidence-based coping mechanisms.
            
            **How it works:**
            *   **Privacy First:** Your sessions are completely private. We use a temporary memory system, meaning your conversations are cleared when your session ends.
            *   **Clinical Literature:** The guidance provided is strictly sourced from established cognitive behavioral techniques and mindfulness practices.
            
            *Disclaimer: Echo Wellness is an automated support tool and is not a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing a crisis, please navigate to the Resources tab to contact a professional.*
            """, elem_classes=["output-text"])

    gr.HTML("<div style='text-align:center;padding:32px;color:#94A3B8;font-size:0.85rem'>Your well-being matters.</div>")
    app.load(fn=on_load, outputs=[history_state, chatbot])

if __name__ == "__main__":
    app.launch()