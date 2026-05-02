import os, re, tempfile
from gtts import gTTS
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import gradio as gr

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY secret not set.")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

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
    Diaphragmatic breathing: breathe into your belly, not chest. Only the belly hand should rise.""",
    """Mindfulness is the practice of non-judgmental present-moment awareness. It reduces rumination, stress, and emotional reactivity.
    Body scan meditation: slowly move attention from feet to head, noticing sensations without judgment.
    Journaling 3 things you are grateful for daily rewires the brain toward positivity and reduces depression over time.""",
    """Crisis resources: If you are in immediate danger, call emergency services (911 in USA, 115 in Pakistan, 999 in UK).
    Pakistan crisis line: Umang helpline 0317-4288665, available Monday-Saturday 3pm-9pm.
    Samaritans UK: 116 123 (free, 24/7). Crisis Text Line US: Text HOME to 741741.""",
    """Panic attacks are intense surges of fear that peak within minutes. They are not dangerous but feel terrifying.
    During a panic attack: remind yourself this will pass, I am safe, this is temporary.
    Progressive muscle relaxation: systematically tense and release muscle groups from toes to face.""",
    """Stress management involves identifying stressors, building coping resources, and creating recovery time.
    Setting boundaries at work and in relationships protects mental energy. Saying no is self-care.
    Nature exposure (even 20 minutes) reduces cortisol.""",
    """Self-care is not selfish. It includes physical, emotional, social, spiritual, and professional dimensions.
    Professional help: therapists, counselors, and psychiatrists provide evidence-based treatment. There is no shame in seeking help."""
]

print("Building RAG...")
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
docs = []
for text in MENTAL_HEALTH_DOCS:
    for chunk in splitter.split_text(text):
        docs.append(Document(page_content=chunk))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("RAG ready!")

client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are Echo, a compassionate, warm mental health companion. "
    "NOT a replacement for therapy. Provide genuine emotional support and coping strategies.\n"
    "- Validate feelings before offering advice\n"
    "- Keep responses concise (2-4 paragraphs)\n"
    "- Use simple, warm language\n"
    "- Encourage professional help when appropriate\n"
    "When crisis detected: immediate validation, grounding technique, "
    "crisis resources (Pakistan: Umang 0317-4288665), encourage reaching out."
)

CRISIS_KEYWORDS = [
    r"\bsuicid", r"\bkill myself\b", r"\bend my life\b", r"\bwant to die\b",
    r"\bno reason to live\b", r"\bhurt myself\b", r"\bself.harm",
    r"\bcant go on\b", r"\bgive up on life\b", r"\bnot worth living\b",
    r"\bwish i was dead\b", r"\bdie by suicide\b"
]

def detect_crisis(text):
    return any(re.search(kw, text.lower()) for kw in CRISIS_KEYWORDS)

def clean_for_api(text):
    return text.encode('ascii', 'ignore').decode('ascii')

def retrieve_context(query):
    results = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in results])

def text_to_speech(text):
    try:
        clean = re.sub(r'[*_#`]', '', text)
        clean = re.sub(r'\s+', ' ', clean).strip()[:800]
        clean = clean.encode('ascii', 'ignore').decode('ascii')
        tts = gTTS(text=clean, lang="en", slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir="/tmp")
        tts.save(tmp.name)
        return tmp.name
    except Exception as e:
        print(f"TTS error: {e}")
        return None

def chat_with_echo(user_message, conversation_history):
    is_crisis = detect_crisis(user_message)
    messages = [{"role": "system", "content": clean_for_api(SYSTEM_PROMPT)}]
    ctx = retrieve_context(user_message)
    if ctx:
        messages.append({"role": "system", "content": clean_for_api("Relevant knowledge:\n" + ctx)})
    if is_crisis:
        messages.append({"role": "system", "content": "CRISIS DETECTED: respond with warmth, grounding, crisis resources (Umang 0317-4288665)."})
    for turn in conversation_history[-16:]:
        messages.append({"role": turn["role"], "content": clean_for_api(turn["content"])})
    messages.append({"role": "user", "content": clean_for_api(user_message)})
    response = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=600, temperature=0.75)
    reply = response.choices[0].message.content
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": reply})
    return reply, conversation_history, is_crisis

MOOD_PROMPTS = {
    "😊 Happy":    "The user is feeling happy. Celebrate with them, ask what's making them happy.",
    "😢 Sad":      "The user is feeling sad. Deep empathy, validate sadness, gently ask what's going on.",
    "😰 Anxious":  "The user is feeling anxious. Validate, offer box breathing (inhale 4, hold 4, exhale 4).",
    "😤 Angry":    "The user is feeling angry. Validate without judgment, help identify what's underneath.",
    "😴 Tired":    "The user is feeling tired. Compassion, ask if physical or emotional tiredness.",
    "😕 Lost":     "The user is feeling lost. Deep understanding, normalize uncertainty as part of growth.",
    "🥰 Grateful": "The user is feeling grateful. Celebrate genuinely, explore what they're grateful for.",
    "😶 Numb":     "The user is feeling numb. Normalize as protective response, suggest gentle grounding.",
}

def format_for_gradio(history):
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
        return history_state, format_for_gradio(history_state), None, ""
    reply, updated, is_crisis = chat_with_echo(message, history_state)
    crisis_msg = "🆘 **Crisis lines:** 🇵🇰 Umang **0317-4288665** · 🇺🇸 **988** · 🇬🇧 **116 123** · Emergency **1122**" if is_crisis else ""
    audio = text_to_speech(reply) if enable_tts else None
    return updated, format_for_gradio(updated), audio, crisis_msg

def on_pill(pill_value, history_state, enable_tts):
    if not pill_value or '::' not in pill_value:
        return history_state, format_for_gradio(history_state), None, ""
    pill_type, value = pill_value.split('::', 1)
    if pill_type == 'mood':
        msg = f"I'm feeling {value.split(' ', 1)[-1] if ' ' in value else value}"
    else:
        tool_map = {
            "🫁 Box Breathing":       "Guide me through box breathing step by step right now.",
            "🌱 5-4-3-2-1 Grounding": "Walk me through the 5-4-3-2-1 grounding technique right now.",
            "🧠 Thought Reframing":   "Help me reframe a negative thought using CBT techniques.",
            "🧘 Body Scan":           "Guide me through a body scan meditation right now.",
            "🙏 Gratitude Practice":  "Guide me through a gratitude practice right now.",
            "💪 Affirmations":        "Give me powerful affirmations for right now.",
        }
        msg = tool_map.get(value, value)
    reply, updated, _ = chat_with_echo(msg, history_state)
    audio = text_to_speech(reply) if enable_tts else None
    return updated, format_for_gradio(updated), audio, ""

def clear_chat():
    return [], [], None, ""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Syne:wght@400;500;600&display=swap');

:root {
    --bg:    #0d0f14;  --bg2: #13161d;  --bg3: #1a1d26;  --bg4: #20242e;
    --border: #2a2d38; --border2: #343844;
    --text: #e8eaf0;   --text2: #9ba3b4; --text3: #555e72;
    --green: #4ade80;  --green2: #22c55e;
    --amber: #fbbf24;  --red: #f87171;   --rdim: #3d1515;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { background: var(--bg) !important; height: 100%; }
.gradio-container {
    background: var(--bg) !important; color: var(--text) !important;
    font-family: 'Syne', system-ui, sans-serif !important;
    max-width: 100% !important; padding: 0 !important; margin: 0 !important;
}
footer, .footer, .svelte-1ax1toq { display: none !important; }
.contain { padding: 0 !important; }
.gap { gap: 0 !important; }
.form, .gr-form, .gr-group { background: transparent !important; border: none !important; box-shadow: none !important; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 99px; }

/* HEADER */
#echo-hdr {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; background: var(--bg2); border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 200;
}
.h-left { display: flex; align-items: center; gap: 9px; }
.h-orb {
    width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
    background: radial-gradient(circle at 35% 30%, #86efac, #4ade80 55%, #15803d);
    font-size: 13px; display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 12px rgba(74,222,128,.45); animation: hglow 3s ease-in-out infinite;
}
@keyframes hglow { 0%,100%{box-shadow:0 0 8px rgba(74,222,128,.3)} 50%{box-shadow:0 0 20px rgba(74,222,128,.6)} }
.h-title { font-family:'Playfair Display',serif !important; font-size:16px !important; font-weight:700 !important; color:var(--text) !important; line-height:1.1 !important; }
.h-sub   { font-size:10px !important; color:var(--text3) !important; margin-top:1px !important; }
.h-badge {
    display:flex; align-items:center; gap:4px; font-size:9px; color:var(--green);
    font-weight:600; letter-spacing:.07em; text-transform:uppercase;
    background:rgba(74,222,128,.07); border:1px solid rgba(74,222,128,.2);
    padding:3px 9px; border-radius:99px;
}
.h-dot { width:4px; height:4px; border-radius:50%; background:var(--green); animation:blink 2.5s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.15} }

/* TABS */
.tabs > .tab-nav { background:var(--bg2) !important; border-bottom:1px solid var(--border) !important; padding:0 12px !important; gap:0 !important; }
.tab-nav button { font-family:'Syne',sans-serif !important; font-size:11px !important; font-weight:500 !important; color:var(--text3) !important; background:transparent !important; border:none !important; border-bottom:2px solid transparent !important; padding:9px 12px !important; border-radius:0 !important; transition:all .2s !important; white-space:nowrap !important; }
.tab-nav button:hover { color:var(--text2) !important; }
.tab-nav button.selected, .tab-nav button[aria-selected="true"] { color:var(--green) !important; border-bottom-color:var(--green) !important; font-weight:600 !important; }

/* CRISIS */
#crisis-out .prose, #crisis-out p { background:rgba(248,113,113,.07) !important; border-bottom:1px solid rgba(248,113,113,.2) !important; padding:8px 14px !important; font-size:12px !important; color:var(--red) !important; margin:0 !important; }
#crisis-out strong { color:var(--red) !important; }
#crisis-out:empty, #crisis-out .prose:empty { display:none !important; padding:0 !important; }

/* PILLS */
.pill-bar { display:flex !important; flex-wrap:wrap !important; gap:5px !important; padding:8px 12px !important; background:var(--bg2) !important; border-bottom:1px solid var(--border) !important; align-items:center !important; }
.pill-bar-label { font-size:9px !important; color:var(--text3) !important; font-weight:600 !important; letter-spacing:.07em !important; text-transform:uppercase !important; margin-right:3px !important; flex-shrink:0 !important; align-self:center !important; }
.echo-pill { display:inline-flex !important; align-items:center !important; background:var(--bg3) !important; border:1px solid var(--border2) !important; color:var(--text2) !important; font-family:'Syne',sans-serif !important; font-size:11px !important; font-weight:500 !important; padding:4px 11px !important; border-radius:99px !important; cursor:pointer !important; transition:all .15s !important; white-space:nowrap !important; line-height:1.4 !important; -webkit-tap-highlight-color:transparent !important; user-select:none !important; }
.echo-pill:hover, .echo-pill:active { background:var(--bg4) !important; border-color:rgba(74,222,128,.4) !important; color:var(--green) !important; }
.echo-pill-tool { background:transparent !important; border-color:var(--border) !important; color:var(--text3) !important; font-size:10px !important; padding:3px 10px !important; }
.echo-pill-tool:hover, .echo-pill-tool:active { background:rgba(74,222,128,.05) !important; border-color:rgba(74,222,128,.3) !important; color:var(--green) !important; }

/* CHATBOT */
#chatbot-main { background:transparent !important; border:none !important; }
#chatbot-main > div { background:transparent !important; border:none !important; }
.message-wrap { padding:4px 12px !important; }
.user .message, .user > div { background:var(--bg3) !important; border:1px solid var(--border2) !important; color:var(--text) !important; border-radius:16px 16px 3px 16px !important; font-size:13px !important; line-height:1.7 !important; padding:10px 14px !important; max-width:80% !important; margin-left:auto !important; font-family:'Syne',sans-serif !important; }
.bot .message, .bot > div { background:transparent !important; border:none !important; color:var(--text) !important; font-size:13px !important; line-height:1.8 !important; padding:8px 2px !important; font-family:'Syne',sans-serif !important; }
.bot .message strong, .bot > div strong { color:var(--green) !important; font-weight:600 !important; }
.bot .message em, .bot > div em { color:var(--amber) !important; }
.bot .message code, .bot > div code { background:rgba(74,222,128,.1) !important; color:var(--green) !important; padding:1px 5px !important; border-radius:4px !important; font-size:11px !important; }
.bot .message ul, .bot .message ol, .bot > div ul, .bot > div ol { padding-left:16px !important; margin:4px 0 !important; }
.bot .message li, .bot > div li { margin-bottom:3px !important; font-size:13px !important; }
.avatar-container { display:none !important; }

/* BOTTOM BAR */
#bottom-bar-row {
    background: var(--bg2) !important;
    border-top: 1px solid var(--border) !important;
    padding: 7px 14px !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}

/* ── TTS CHECKBOX styled as a pill toggle ── */
#tts-checkbox {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    min-width: unset !important;
}
#tts-checkbox label {
    display: flex !important;
    align-items: center !important;
    gap: 7px !important;
    cursor: pointer !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 11px !important;
    color: var(--text3) !important;
    user-select: none !important;
    white-space: nowrap !important;
}
/* Replace native checkbox with CSS toggle track */
#tts-checkbox input[type="checkbox"] {
    -webkit-appearance: none !important;
    appearance: none !important;
    width: 34px !important;
    height: 18px !important;
    background: var(--border2) !important;
    border-radius: 99px !important;
    border: none !important;
    outline: none !important;
    cursor: pointer !important;
    position: relative !important;
    flex-shrink: 0 !important;
    transition: background 0.2s !important;
    margin: 0 !important;
}
#tts-checkbox input[type="checkbox"]::after {
    content: '' !important;
    display: block !important;
    width: 14px !important;
    height: 14px !important;
    background: #fff !important;
    border-radius: 50% !important;
    position: absolute !important;
    top: 2px !important;
    left: 2px !important;
    transition: left 0.2s !important;
    pointer-events: none !important;
}
#tts-checkbox input[type="checkbox"]:checked {
    background: var(--green) !important;
}
#tts-checkbox input[type="checkbox"]:checked::after {
    left: 18px !important;
}

/* CLEAR BUTTON */
#clear-btn {
    margin-left: auto !important;
}
#clear-btn button {
    background: transparent !important;
    border: 1px solid rgba(248,113,113,0.3) !important;
    color: var(--red) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 11px !important;
    padding: 4px 14px !important;
    border-radius: 99px !important;
    cursor: pointer !important;
    box-shadow: none !important;
    transition: all 0.15s !important;
}
#clear-btn button:hover {
    background: var(--rdim) !important;
    border-color: var(--red) !important;
}

/* INPUT */
#input-area { padding:8px 12px 10px !important; background:var(--bg) !important; border-top:1px solid var(--border) !important; }
#input-inner { display:flex !important; align-items:flex-end !important; gap:7px !important; background:var(--bg2) !important; border:1.5px solid var(--border2) !important; border-radius:14px !important; padding:5px 5px 5px 14px !important; transition:border-color .2s,box-shadow .2s !important; }
#input-inner:focus-within { border-color:rgba(74,222,128,.45) !important; box-shadow:0 0 0 3px rgba(74,222,128,.07) !important; }
#msg-input { flex:1 !important; background:transparent !important; border:none !important; box-shadow:none !important; outline:none !important; }
#msg-input textarea, #msg-input input { background:transparent !important; border:none !important; box-shadow:none !important; outline:none !important; color:var(--text) !important; font-family:'Syne',sans-serif !important; font-size:14px !important; line-height:1.5 !important; resize:none !important; padding:6px 0 !important; min-height:36px !important; max-height:120px !important; }
#msg-input textarea::placeholder { color:var(--text3) !important; }
#send-btn { width:34px !important; height:34px !important; min-width:34px !important; border-radius:50% !important; background:var(--green) !important; border:none !important; color:#071a0f !important; font-size:16px !important; font-weight:900 !important; display:flex !important; align-items:center !important; justify-content:center !important; cursor:pointer !important; flex-shrink:0 !important; transition:all .18s !important; padding:0 !important; margin-bottom:1px !important; line-height:1 !important; }
#send-btn:hover { background:var(--green2) !important; transform:scale(1.08) !important; }
#send-btn:disabled { opacity:.25 !important; transform:none !important; }
.input-hint { text-align:center !important; font-size:10px !important; color:var(--text3) !important; margin-top:6px !important; line-height:1.5 !important; font-family:'Syne',sans-serif !important; }
#audio-out { display:none !important; }

/* RESOURCES */
.res-wrap { padding:12px; }
.res-card { background:var(--bg2); border:1px solid var(--border); border-radius:12px; padding:12px 14px; margin-bottom:10px; }
.res-card h3 { font-family:'Playfair Display',serif !important; font-size:13px !important; margin-bottom:9px !important; }
.res-row { display:flex; justify-content:space-between; align-items:center; padding:6px 9px; background:var(--bg3); border-radius:7px; margin-bottom:4px; border:1px solid var(--border); }
.res-name { font-size:12px !important; font-weight:600 !important; color:var(--text) !important; }
.res-desc { font-size:10px !important; color:var(--text3) !important; }
.res-num  { font-size:11px !important; font-weight:700 !important; flex-shrink:0; margin-left:8px; }
"""

RESOURCES_HTML = """<div class="res-wrap">
  <div class="res-card">
    <h3 style="color:#f87171">🇵🇰 Pakistan</h3>
    <div class="res-row"><div><div class="res-name">Emergency Rescue</div><div class="res-desc">Police / Ambulance</div></div><div class="res-num" style="color:#f87171;font-size:17px">1122</div></div>
    <div class="res-row"><div><div class="res-name">Umang Helpline</div><div class="res-desc">Mental Health 24/7</div></div><div class="res-num" style="color:#4ade80">0317-4288665</div></div>
    <div class="res-row"><div><div class="res-name">Taskeen</div><div class="res-desc">Psychological support</div></div><div class="res-num" style="color:#4ade80">0316-8275336</div></div>
    <div class="res-row"><div><div class="res-name">Rozan Counselling</div><div class="res-desc">Islamabad</div></div><div class="res-num" style="color:#4ade80">051-2890505</div></div>
    <div class="res-row"><div><div class="res-name">Edhi Foundation</div><div class="res-desc">Crisis & rescue</div></div><div class="res-num" style="color:#f87171;font-size:17px">115</div></div>
  </div>
  <div class="res-card">
    <h3 style="color:#4ade80">🌍 International</h3>
    <div class="res-row"><div><div class="res-name">USA / Canada</div><div class="res-desc">Suicide & Crisis Lifeline</div></div><div class="res-num" style="color:#4ade80;font-size:17px">988</div></div>
    <div class="res-row"><div><div class="res-name">UK — Samaritans</div><div class="res-desc">Free 24/7</div></div><div class="res-num" style="color:#4ade80">116 123</div></div>
    <div class="res-row"><div><div class="res-name">Australia Lifeline</div></div><div class="res-num" style="color:#4ade80">13 11 14</div></div>
    <div class="res-row"><div><div class="res-name">India — iCall</div><div class="res-desc">TISS helpline</div></div><div class="res-num" style="color:#4ade80">9152987821</div></div>
    <div class="res-row"><div><div class="res-name">Crisis Text (US)</div><div class="res-desc">Text HOME → 741741</div></div><div class="res-num">💬</div></div>
    <div class="res-row"><div><div class="res-name">Global Directory</div><div class="res-desc">findahelpline.com</div></div><div class="res-num">🌐</div></div>
  </div>
  <div style="text-align:center;font-size:10px;color:#555e72;padding:4px 0 8px;line-height:1.7">
    ⚠️ Echo is not a substitute for professional help.<br>
    Immediate danger → <strong style="color:#f87171">1122</strong> (PK) · <strong style="color:#f87171">911</strong> (US) · <strong style="color:#f87171">999</strong> (UK)
  </div>
</div>"""

MOOD_BAR_HTML = """
<div class="pill-bar">
  <span class="pill-bar-label">Mood</span>
  <button class="echo-pill" onclick="echoPill('mood','😊 Happy')">😊 Happy</button>
  <button class="echo-pill" onclick="echoPill('mood','😢 Sad')">😢 Sad</button>
  <button class="echo-pill" onclick="echoPill('mood','😰 Anxious')">😰 Anxious</button>
  <button class="echo-pill" onclick="echoPill('mood','😤 Angry')">😤 Angry</button>
  <button class="echo-pill" onclick="echoPill('mood','😴 Tired')">😴 Tired</button>
  <button class="echo-pill" onclick="echoPill('mood','😕 Lost')">😕 Lost</button>
  <button class="echo-pill" onclick="echoPill('mood','🥰 Grateful')">🥰 Grateful</button>
  <button class="echo-pill" onclick="echoPill('mood','😶 Numb')">😶 Numb</button>
</div>"""

TOOL_BAR_HTML = """
<div class="pill-bar">
  <span class="pill-bar-label">Tools</span>
  <button class="echo-pill echo-pill-tool" onclick="echoPill('tool','🫁 Box Breathing')">🫁 Box Breathing</button>
  <button class="echo-pill echo-pill-tool" onclick="echoPill('tool','🌱 5-4-3-2-1 Grounding')">🌱 Grounding</button>
  <button class="echo-pill echo-pill-tool" onclick="echoPill('tool','🧠 Thought Reframing')">🧠 Reframing</button>
  <button class="echo-pill echo-pill-tool" onclick="echoPill('tool','🧘 Body Scan')">🧘 Body Scan</button>
  <button class="echo-pill echo-pill-tool" onclick="echoPill('tool','🙏 Gratitude Practice')">🙏 Gratitude</button>
  <button class="echo-pill echo-pill-tool" onclick="echoPill('tool','💪 Affirmations')">💪 Affirmations</button>
</div>"""

# Minimal JS — only pills, no TTS hackery needed
BRIDGE_JS = """
<script>
function echoPill(type, value) {
    var box = document.querySelector('#pill-bridge textarea');
    if (!box) return;
    var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(box, type + '::' + value);
    box.dispatchEvent(new Event('input', { bubbles: true }));
    setTimeout(function() {
        var btn = document.querySelector('#pill-send-btn');
        if (btn) btn.click();
    }, 80);
}
</script>
"""

with gr.Blocks(theme=gr.themes.Base(), css=CSS, title="Echo — Wellness") as app:
    history_state = gr.State([])

    gr.HTML(BRIDGE_JS)

    gr.HTML("""<div id="echo-hdr">
      <div class="h-left">
        <div class="h-orb">🌿</div>
        <div>
          <div class="h-title">Echo</div>
          <div class="h-sub">Mental Wellness Companion</div>
        </div>
      </div>
      <div class="h-badge"><span class="h-dot"></span>Private</div>
    </div>""")

    with gr.Tabs():

        with gr.Tab("💬 Chat"):

            crisis_out = gr.Markdown("", elem_id="crisis-out")
            gr.HTML(MOOD_BAR_HTML)
            gr.HTML(TOOL_BAR_HTML)

            chatbot = gr.Chatbot(
                label="", elem_id="chatbot-main",
                type="messages", show_label=False,
                height=400,
                placeholder=(
                    "<div style='text-align:center;padding:50px 16px'>"
                    "<div style='font-size:34px;margin-bottom:10px'>🌿</div>"
                    "<div style='font-family:Playfair Display,serif;font-size:16px;"
                    "color:#9ba3b4;margin-bottom:7px'>Hello, I'm Echo</div>"
                    "<div style='font-size:12px;color:#555e72;line-height:1.8;max-width:260px;margin:0 auto'>"
                    "A safe space to share what's on your mind.<br>"
                    "I'm here to listen — without judgment.</div></div>"
                ),
            )

            audio_out = gr.Audio(label="", autoplay=True, elem_id="audio-out", visible=False)

            # ── BOTTOM BAR ──
            # tts_checkbox is a real native Gradio Checkbox — no JS sync needed.
            # CSS above gives it the visual toggle appearance.
            with gr.Row(elem_id="bottom-bar-row"):
                tts_checkbox = gr.Checkbox(
                    value=False,
                    label="🔊 Voice",
                    elem_id="tts-checkbox",
                    container=False,
                    scale=0,
                    min_width=100,
                )
                clear_btn = gr.Button(
                    "🗑 Clear",
                    elem_id="clear-btn",
                    scale=0,
                    min_width=80,
                )

            # Hidden pill bridge
            with gr.Row(visible=False):
                pill_bridge = gr.Textbox(value="", elem_id="pill-bridge")
                pill_send   = gr.Button("send", elem_id="pill-send-btn")

            # Input area
            with gr.Column(elem_id="input-area"):
                with gr.Row(elem_id="input-inner"):
                    msg_input = gr.Textbox(
                        placeholder="Message Echo…",
                        show_label=False, container=False,
                        lines=1, max_lines=5,
                        elem_id="msg-input", scale=9, autofocus=True,
                    )
                    send_btn = gr.Button("↑", elem_id="send-btn", scale=0, min_width=38)

                gr.HTML("""<div class="input-hint">
                    Echo is not a substitute for professional care &nbsp;·&nbsp;
                    Crisis: <strong style="color:#555e72">1122</strong> (PK)
                    · <strong style="color:#555e72">988</strong> (US)
                    · <strong style="color:#555e72">116 123</strong> (UK)
                </div>""")

        with gr.Tab("📞 Resources"):
            gr.HTML(RESOURCES_HTML)

    # ── WIRING ──
    send_btn.click(
        fn=on_send,
        inputs=[msg_input, history_state, tts_checkbox],
        outputs=[history_state, chatbot, audio_out, crisis_out]
    ).then(lambda: "", outputs=[msg_input])

    msg_input.submit(
        fn=on_send,
        inputs=[msg_input, history_state, tts_checkbox],
        outputs=[history_state, chatbot, audio_out, crisis_out]
    ).then(lambda: "", outputs=[msg_input])

    pill_send.click(
        fn=on_pill,
        inputs=[pill_bridge, history_state, tts_checkbox],
        outputs=[history_state, chatbot, audio_out, crisis_out]
    ).then(lambda: "", outputs=[pill_bridge])

    clear_btn.click(
        fn=clear_chat,
        outputs=[history_state, chatbot, audio_out, crisis_out]
    )

if __name__ == "__main__":
    app.launch()
