"""
Streamlit Dashboard — Krushi Node AI Chatbot
- Suggestion chips below input (like real chatbots)
- All chips/quick-actions go through full pipeline so F1 triggers
- Fixed open_timeout and WebSocket error handling
"""

import html
import streamlit as st
import asyncio
import websockets
import json
import base64
import uuid
from datetime import datetime
from audio_recorder_streamlit import audio_recorder

st.set_page_config(
    page_title="Krushi Node — AI Chatbot",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── user bubble ── */
.msg-user {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    margin: 10px 0;
}
.msg-user .bubble {
    background: #1e3a5f;
    color: #e8f4ff !important;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 18px;
    max-width: 72%;
    font-size: 1rem;
    line-height: 1.6;
    word-break: break-word;
}
.msg-user .meta {
    font-size: 0.72rem;
    color: #8ab4d4 !important;
    margin-top: 4px;
    padding-right: 4px;
}

/* ── assistant bubble ── */
.msg-assistant {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    margin: 10px 0;
}
.msg-assistant .label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #4caf50 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.msg-assistant .bubble {
    background: #1a2e1a;
    color: #d4f4d4 !important;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 18px;
    max-width: 82%;
    font-size: 1rem;
    line-height: 1.7;
    word-break: break-word;
    border-left: 3px solid #4caf50;
}
.msg-assistant .meta {
    font-size: 0.72rem;
    color: #6b8f6b !important;
    margin-top: 4px;
    padding-left: 4px;
}

/* ── clarification history bubble ── */
.msg-clarification {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    margin: 10px 0;
}
.msg-clarification .label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #e6a817 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}
.msg-clarification .bubble {
    background: #2e2200;
    color: #ffe8a0 !important;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 18px;
    max-width: 82%;
    font-size: 1rem;
    line-height: 1.6;
    border-left: 3px solid #e6a817;
    word-break: break-word;
}
.msg-clarification .meta {
    font-size: 0.72rem;
    color: #9e7b00 !important;
    margin-top: 4px;
    padding-left: 4px;
}

/* ── live clarification panel ── */
.clr-panel {
    background: #1c1400;
    border: 1px solid #e6a817;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0 8px 0;
}
.clr-panel .clr-heading {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffe066 !important;
    margin-bottom: 4px;
}
.clr-panel .clr-subtext {
    font-size: 0.85rem;
    color: #c9a843 !important;
    margin-bottom: 0;
}

/* ── suggestion chips ── */
.chip-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #aaa !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

/* divider */
.section-divider {
    border: none;
    border-top: 1px solid #2a2a2a;
    margin: 16px 0;
}

/* Make ALL stButtons consistent */
div[data-testid="stButton"] > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}

/* Clarification option buttons — amber style */
.clr-opts div[data-testid="stButton"] > button {
    background: #2a1e00 !important;
    color: #ffe8a0 !important;
    border: 1px solid #c8900a !important;
    font-size: 0.9rem !important;
    padding: 10px 8px !important;
}
.clr-opts div[data-testid="stButton"] > button:hover {
    background: #3d2d00 !important;
    border-color: #ffbb33 !important;
}

/* Suggestion chip buttons — small, muted */
.chip-row div[data-testid="stButton"] > button {
    background: #1e2a1e !important;
    color: #b8d4b8 !important;
    border: 1px solid #3a5a3a !important;
    font-size: 0.82rem !important;
    padding: 6px 10px !important;
    height: auto !important;
}
.chip-row div[data-testid="stButton"] > button:hover {
    background: #2a3d2a !important;
    color: #d4f4d4 !important;
    border-color: #5a8a5a !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "chat_history":          [],
        "ws_url":                "ws://localhost:8002/ws/chat",
        "processing":            False,
        "session_id":            str(uuid.uuid4()),
        "pending_clarification": None,
        "chosen_intent_key":     None,
        # Stores a message that was clicked (chip/quick-action) and needs to be sent
        "pending_send":          None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def _esc(text: str) -> str:
    """HTML-escape + newline → <br> so LLM output never breaks a div."""
    return html.escape(str(text)).replace("\n", "<br>")


# ── WebSocket helpers ─────────────────────────────────────────────────────────

async def _ws_connect():
    """
    Create WebSocket connection.
    open_timeout=None means wait as long as needed for the handshake
    (default was 10s which caused 'timed out during opening handshake'
    when the server was slow to respond).
    """
    return await websockets.connect(
        st.session_state.ws_url,
        ping_interval=20,
        ping_timeout=120,
        max_size=10_000_000,
        compression=None,
        open_timeout=None,          # ← removed 10s hard cap
    )


async def send_text(text: str, session_id: str):
    """
    Send a text_input message.
    Returns (answer_str_or_None, audio_data, clarification_dict_or_None).
    """
    try:
        async with await _ws_connect() as ws:
            await ws.send(json.dumps({
                "type":       "text_input",
                "text":       text,
                "session_id": session_id,
            }))
            full, audio, clr = "", None, None
            while True:
                try:
                    data = json.loads(await asyncio.wait_for(ws.recv(), timeout=120.0))
                    t    = data.get("type")
                    if t == "text_output":
                        full += data.get("text", "")
                        if data.get("is_complete"):
                            break
                    elif t == "clarification_request":
                        clr = data
                        break
                    elif t == "audio_output":
                        audio = data.get("audio_data")
                        if data.get("is_complete"):
                            break
                    elif t == "error":
                        full = f"Error: {data.get('error', '')}"
                        break
                    elif t in ("status", "kshop_products"):
                        continue
                except asyncio.TimeoutError:
                    break
            return full or None, audio, clr
    except (OSError, ConnectionRefusedError):
        return "⚠️ Backend server is not running. Please start it with: python app/main.py", None, None
    except Exception as e:
        return f"Connection error: {e}", None, None


async def send_clarification(intent_key: str, session_id: str):
    """Send clarification_response, return (answer, audio)."""
    try:
        async with await _ws_connect() as ws:
            await ws.send(json.dumps({
                "type":       "clarification_response",
                "session_id": session_id,
                "intent_key": intent_key,
            }))
            full, audio = "", None
            while True:
                try:
                    data = json.loads(await asyncio.wait_for(ws.recv(), timeout=120.0))
                    t    = data.get("type")
                    if t == "text_output":
                        full += data.get("text", "")
                        if data.get("is_complete"):
                            break
                    elif t == "audio_output":
                        audio = data.get("audio_data")
                        if data.get("is_complete"):
                            break
                    elif t == "error":
                        full = f"Error: {data.get('error', '')}"
                        break
                    elif t in ("status", "kshop_products"):
                        continue
                except asyncio.TimeoutError:
                    break
            return full or "No answer returned. Please try again.", audio
    except (OSError, ConnectionRefusedError):
        return "⚠️ Backend server is not running. Please start it with: python app/main.py", None
    except Exception as e:
        return f"Connection error: {e}", None


async def send_audio(audio_bytes: bytes, session_id: str):
    try:
        async with await _ws_connect() as ws:
            await ws.send(json.dumps({
                "type":         "audio_input",
                "audio_data":   base64.b64encode(audio_bytes).decode(),
                "audio_format": "wav",
                "session_id":   session_id,
            }))
            full, audio = "", None
            while True:
                try:
                    data = json.loads(await asyncio.wait_for(ws.recv(), timeout=120.0))
                    t    = data.get("type")
                    if t == "text_output":
                        full += data.get("text", "")
                        if data.get("is_complete"):
                            break
                    elif t == "audio_output":
                        audio = data.get("audio_data")
                        if data.get("is_complete"):
                            break
                    elif t == "error":
                        full = f"Error: {data.get('error', '')}"
                        break
                except asyncio.TimeoutError:
                    break
            return full or "No answer returned.", audio
    except (OSError, ConnectionRefusedError):
        return "⚠️ Backend server is not running. Please start it with: python app/main.py", None
    except Exception as e:
        return f"Connection error: {e}", None


# ── Core send pipeline — used by text input, chips, and quick actions ─────────

def execute_send(query: str):
    """
    Full send pipeline for ANY query (text input, chip click, quick action).
    Adds user bubble, calls backend, handles F1 clarification or answer.
    """
    st.session_state.processing = True
    st.session_state.chat_history.append(
        {"role": "user", "content": query, "timestamp": _ts()}
    )
    with st.spinner("🤔 Thinking..."):
        ans, aud, clr = asyncio.run(
            send_text(query, st.session_state.session_id)
        )
    if clr:
        # F1 triggered — show live buttons only (do NOT add to chat_history here).
        # Once the user picks an option, their choice + the answer are added instead.
        st.session_state.pending_clarification = clr
    else:
        st.session_state.chat_history.append(
            {"role": "assistant",
             "content": ans or "No answer returned. Please try again.",
             "timestamp": _ts(), "audio_data": aud}
        )
    st.session_state.processing = False


# ── Suggestion chips definition ───────────────────────────────────────────────
# These go through the full pipeline — ambiguous ones will trigger F1.
# Grouped by category for display.

SUGGESTION_CHIPS = [
    # Row 1 — crops (all ambiguous → will trigger F1)
    ("🌾 Wheat",          "wheat"),
    ("🌱 Cotton",         "kapas"),
    ("🌽 Bajra",          "bajra"),
    ("🥜 Groundnut",      "magfali"),
    ("🧅 Onion",          "onion"),
    # Row 2 — direct intent (won't trigger F1)
    ("📈 Kapas Bhav",     "kapas bhav today"),
    ("🛒 K-Shop Products","show me kshop products"),
    ("📦 Buy/Sell",       "show me buy sell listings"),
    ("📰 Farm News",      "show me latest agricultural news"),
    ("🎬 Farm Videos",    "show me farming videos"),
]


# ── Message renderers ─────────────────────────────────────────────────────────

def render_user(content: str, ts: str):
    st.markdown(f"""
    <div class="msg-user">
        <div class="bubble">{_esc(content)}</div>
        <div class="meta">🧑 You · {ts}</div>
    </div>""", unsafe_allow_html=True)


def render_assistant(content: str, ts: str, audio_data=None):
    st.markdown(f"""
    <div class="msg-assistant">
        <div class="label">🤖 Assistant</div>
        <div class="bubble">{_esc(content)}</div>
        <div class="meta">{ts}</div>
    </div>""", unsafe_allow_html=True)
    if audio_data:
        try:
            b64 = base64.b64encode(base64.b64decode(audio_data)).decode()
            st.markdown(
                f'<audio autoplay>'
                f'<source src="data:audio/wav;base64,{b64}" type="audio/wav">'
                f'</audio>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass


def render_clarification_history(content: str, ts: str):
    st.markdown(f"""
    <div class="msg-clarification">
        <div class="label">🤔 Clarification needed</div>
        <div class="bubble">{_esc(content)}</div>
        <div class="meta">{ts}</div>
    </div>""", unsafe_allow_html=True)


def render_live_clarification(clr: dict):
    """
    Live clarification panel — shown only while pipeline is paused.
    Disappears once user picks an option.
    """
    question = clr.get("question", "Please select an option:")
    options  = clr.get("options", [])

    st.markdown(f"""
    <div class="clr-panel">
        <div class="clr-heading">🤔 {_esc(question)}</div>
        <div class="clr-subtext">Tap one of the options below:</div>
    </div>""", unsafe_allow_html=True)

    n    = min(len(options), 3)
    cols = st.columns(n) if n > 0 else [st.container()]

    st.markdown('<div class="clr-opts">', unsafe_allow_html=True)
    for i, opt in enumerate(options):
        with cols[i % n]:
            if st.button(
                opt.get("label", ""),
                key=f"clr_{opt.get('intent_key', i)}_{id(clr)}",
                use_container_width=True,
            ):
                st.session_state.chosen_intent_key = opt.get("intent_key")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ── Layout ────────────────────────────────────────────────────────────────────

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Server Status")
    try:
        import requests as _req
        r = _req.get("http://localhost:8002/health", timeout=3)
        if r.status_code == 200:
            st.success("✅ Connected")
        else:
            st.error("❌ Disconnected")
    except Exception:
        st.error("❌ Server not running")

    st.divider()

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history          = []
        st.session_state.pending_clarification = None
        st.session_state.session_id            = str(uuid.uuid4())
        st.session_state.pending_send          = None
        st.rerun()

    st.divider()
    st.metric("Messages", len(st.session_state.chat_history))

# Title
st.markdown("# 🌾 Krushi Node — AI Chatbot")
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    role = msg["role"]
    if role == "user":
        render_user(msg["content"], msg["timestamp"])
    else:
        render_assistant(msg["content"], msg["timestamp"], msg.get("audio_data"))

# ── STEP A: Consume pending_send (chip / quick-action click) ─────────────────
# When a chip or quick-action button is clicked, it stores the query in
# pending_send and calls st.rerun(). On the next run this block fires,
# executes the full send pipeline (so F1 can trigger), then reruns again
# to show the result.

if st.session_state.pending_send and not st.session_state.processing:
    query = st.session_state.pending_send
    st.session_state.pending_send = None
    execute_send(query)
    st.rerun()

# ── STEP B: Consume clarification button tap ──────────────────────────────────

if st.session_state.chosen_intent_key:
    chosen = st.session_state.chosen_intent_key
    st.session_state.chosen_intent_key = None

    label_map = {
        o["intent_key"]: o["label"]
        for o in (st.session_state.pending_clarification or {}).get("options", [])
    }
    chosen_label = label_map.get(chosen, chosen)

    st.session_state.chat_history.append(
        {"role": "user", "content": chosen_label, "timestamp": _ts()}
    )
    st.session_state.pending_clarification = None

    with st.spinner("🤔 Getting your answer..."):
        ans, aud = asyncio.run(
            send_clarification(chosen, st.session_state.session_id)
        )
    st.session_state.chat_history.append(
        {"role": "assistant",
         "content": ans or "No answer returned. Please try again.",
         "timestamp": _ts(), "audio_data": aud}
    )
    st.rerun()

# ── STEP C: Show live clarification buttons ───────────────────────────────────

if st.session_state.pending_clarification:
    render_live_clarification(st.session_state.pending_clarification)

# ── Text Input ────────────────────────────────────────────────────────────────

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("### ✍️ Type your message")

col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        "msg", key="user_input",
        placeholder="e.g. wheat price, kapas bhav, kshop products...",
        label_visibility="collapsed",
    )
with col2:
    send_btn = st.button("Send ➤", type="primary", use_container_width=True)

if send_btn and user_input and not st.session_state.processing:
    execute_send(user_input)
    st.rerun()

# ── Suggestion Chips ──────────────────────────────────────────────────────────
# Like a real chatbot — clickable chips the user can tap instead of typing.
# Each chip goes through the full pipeline, so ambiguous ones trigger F1.

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="chip-label">💡 Tap a suggestion</div>', unsafe_allow_html=True)

# Row 1 — crop / ambiguous queries (will trigger F1 confirmation)
st.markdown('<div class="chip-row">', unsafe_allow_html=True)
row1 = st.columns(5)
for i, (label, query) in enumerate(SUGGESTION_CHIPS[:5]):
    with row1[i]:
        if st.button(label, key=f"chip_r1_{i}", use_container_width=True):
            if not st.session_state.processing:
                st.session_state.pending_send = query
                st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Row 2 — direct intent queries (won't trigger F1, go straight to answer)
st.markdown('<div class="chip-row">', unsafe_allow_html=True)
row2 = st.columns(5)
for i, (label, query) in enumerate(SUGGESTION_CHIPS[5:]):
    with row2[i]:
        if st.button(label, key=f"chip_r2_{i}", use_container_width=True):
            if not st.session_state.processing:
                st.session_state.pending_send = query
                st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ── Voice Input ───────────────────────────────────────────────────────────────

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("### 🎤 Voice Input")
st.caption("Click the microphone, speak, then wait for the response.")

audio_bytes = audio_recorder(
    text="", recording_color="#e74c3c", neutral_color="#3498db",
    icon_name="microphone", icon_size="2x",
    pause_threshold=2.0, sample_rate=16000,
)

if audio_bytes and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.chat_history.append(
        {"role": "user", "content": "🎤 Voice message sent", "timestamp": _ts()}
    )
    with st.spinner("🎙️ Processing voice..."):
        ans, aud = asyncio.run(
            send_audio(audio_bytes, st.session_state.session_id)
        )
    st.session_state.chat_history.append(
        {"role": "assistant", "content": ans, "timestamp": _ts(), "audio_data": aud}
    )
    st.session_state.processing = False
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.78rem;padding:8px 0'>"
    "Krushi Node · AI Chatbot v1.0 · FastAPI · Groq · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
