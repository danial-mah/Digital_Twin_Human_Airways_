"""
pages/9_Ask_AI.py
==================
AI Virtual Assistant — animated avatar (4 states), browser Web Speech STT,
browser speechSynthesis TTS, and the same offline RAG backend.

Voice input:  browser Web Speech API (Chrome/Edge, free, no key)
Voice output: browser speechSynthesis (all modern browsers, offline)
"""

import sys
import re
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rag import (
    get_knowledge_base, compose_local_answer,
    generate_answer, _get_api_key, _HAS_SEMANTIC,
)
from core.i18n import t, lang_selector

st.set_page_config(page_title="Ask AI", page_icon="🤖", layout="wide")
lang_selector()

# ── Language-aware constants ───────────────────────────────────────────────────
def _suggested():
    return [t(f"ask_q{i}_en") for i in range(1, 9)]

SUGGESTED_KEYS = ["ask_q1_en","ask_q2_en","ask_q3_en","ask_q4_en",
                   "ask_q5_en","ask_q6_en","ask_q7_en","ask_q8_en"]

STATE_COLORS = {
    "idle":      "#4EB3D3",
    "listening": "#4CAF50",
    "thinking":  "#F4A261",
    "speaking":  "#9C27B0",
}
STATE_LABELS = {
    "idle":      "Ready",
    "listening": "Listening…",
    "thinking":  "Thinking…",
    "speaking":  "Speaking…",
}

# ── Session state ──────────────────────────────────────────────────────────────
for _k, _v in [
    ("kb",             None),
    ("chat_history",   []),
    ("tts_enabled",    True),
    ("avatar_state",   "idle"),
    ("answer_counter", 0),
    ("answer_tts",     ""),
    ("prefill",        ""),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.kb is None:
    with st.spinner("Loading knowledge base…"):
        st.session_state.kb = get_knowledge_base()

kb = st.session_state.kb


# ── Avatar HTML/Canvas (4 animated states) ─────────────────────────────────────
def _avatar_html(state: str) -> str:
    color = STATE_COLORS.get(state, "#4EB3D3")
    label = STATE_LABELS.get(state, "Ready")
    uid   = f"av{int(time.monotonic() * 1e6) % 10_000_000}"
    return f"""
<div style="text-align:center;padding:6px 0 4px;">
<canvas id="{uid}" width="160" height="160"
        style="display:block;margin:0 auto;"></canvas>
<div style="margin-top:7px;">
  <span style="
    background:{color}1a;color:{color};padding:4px 14px;
    border-radius:14px;font-size:11px;font-weight:bold;
    border:1px solid {color}40;letter-spacing:.4px;
  ">{label}</span>
</div>
</div>
<script>
(function(){{
  const C=document.getElementById('{uid}');
  if(!C)return;
  const ctx=C.getContext('2d');
  const cx=C.width/2,cy=C.height/2,R=Math.min(cx,cy)-8;
  const STATE='{state}';
  function hexRgb(h){{
    const r=/^#?([a-f\d]{{2}})([a-f\d]{{2}})([a-f\d]{{2}})$/i.exec(h);
    return r?[parseInt(r[1],16),parseInt(r[2],16),parseInt(r[3],16)]:[78,179,211];
  }}
  const [cr,cg,cb]=hexRgb('{color}');
  function ca(a){{return`rgba(${{cr}},${{cg}},${{cb}},${{a}})`;}}
  let ph=0;
  function frame(){{
    ctx.clearRect(0,0,C.width,C.height);
    // background disc
    ctx.beginPath();ctx.arc(cx,cy,R,0,2*Math.PI);ctx.fillStyle='#1a1d26';ctx.fill();
    if(STATE==='idle'){{
      // two rings + AI text
      ctx.lineWidth=2;ctx.strokeStyle='{color}';
      ctx.beginPath();ctx.arc(cx,cy,R-3,0,2*Math.PI);ctx.stroke();
      ctx.beginPath();ctx.arc(cx,cy,R/2,0,2*Math.PI);ctx.stroke();
      ctx.beginPath();ctx.arc(cx,cy,R/3,0,2*Math.PI);ctx.fillStyle=ca(0.22);ctx.fill();
      ctx.fillStyle='{color}';ctx.font='bold 20px Segoe UI,Arial';
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('AI',cx,cy);
    }} else if(STATE==='listening'){{
      // expanding rings
      for(let i=0;i<3;i++){{
        const t=(ph+i/3)%1,r=R*.25+R*.75*t,a=(1-t)*.85;
        ctx.beginPath();ctx.arc(cx,cy,r,0,2*Math.PI);
        ctx.strokeStyle=ca(a);ctx.lineWidth=2;ctx.stroke();
      }}
      ctx.beginPath();ctx.arc(cx,cy,R*.28,0,2*Math.PI);ctx.fillStyle='{color}';ctx.fill();
      ctx.font='22px Arial';ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText('🎙',cx,cy+1);
    }} else if(STATE==='thinking'){{
      // three orbiting dots
      const or=R*.55,dr=R*.13;
      for(let i=0;i<3;i++){{
        const ang=ph*2*Math.PI+i*2*Math.PI/3;
        const dx=cx+or*Math.cos(ang),dy=cy+or*Math.sin(ang);
        const br=.4+.6*((Math.sin(ang+ph*Math.PI)+1)/2);
        ctx.beginPath();ctx.arc(dx,dy,dr,0,2*Math.PI);ctx.fillStyle=ca(br);ctx.fill();
      }}
      ctx.beginPath();ctx.arc(cx,cy,R*.22,0,2*Math.PI);ctx.fillStyle=ca(.45);ctx.fill();
    }} else if(STATE==='speaking'){{
      // waveform bars
      const nB=7,bW=Math.max(4,R*1.1/nB-2),tW=nB*(bW+2),x0=cx-tW/2;
      for(let i=0;i<nB;i++){{
        const t=ph*2*Math.PI+i*.7,hf=.25+.65*Math.abs(Math.sin(t)),bH=R*hf;
        const bx=x0+i*(bW+2),by=cy-bH/2,al=.65+.35*Math.abs(Math.sin(t));
        ctx.fillStyle=ca(al);ctx.fillRect(bx,by,bW,bH);
      }}
    }}
    ph=(ph+.035)%1;
  }}
  setInterval(frame,40);
}})();
</script>
"""


# ── TTS injector (height-0 iframe, speaks once per answer) ────────────────────
def _tts_html(text: str, answer_id: int, lang: str = "en") -> str:
    safe = (text.replace("\\", "\\\\")
                .replace("`", "\\`")
                .replace("$", "\\$")
                .replace("\n", " ")
                .replace('"', '\\"'))
    return f"""
<script>
(function(){{
  const id={answer_id};
  const prev=parseInt(sessionStorage.getItem('ha_tts_id')||'0');
  if(id<=prev)return;
  sessionStorage.setItem('ha_tts_id',String(id));
  if(!('speechSynthesis' in window))return;
  window.speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(`{safe}`);
  u.rate=1.05;u.pitch=1.0;u.volume=0.92;
  // prefer a natural-sounding voice matching the UI language
  const voices=window.speechSynthesis.getVoices();
  const pref=voices.find(v=>v.lang.startsWith('{lang}')&&v.localService)
            ||voices.find(v=>v.lang.startsWith('{lang}'));
  if(pref)u.voice=pref;
  window.speechSynthesis.speak(u);
}})();
</script>
"""


# ── Voice-input component (Web Speech API -> fills Streamlit chat_input) ────────
_VOICE_HTML_TMPL = (
    '<div style="display:flex;align-items:center;gap:10px;padding:2px 0;">'
    '  <button id="micBtn" onclick="toggleMic()" style="'
    '    background:#252836;border:1px solid #3a3d4e;border-radius:6px;'
    '    padding:7px 16px;font-size:13px;color:#ccc;cursor:pointer;'
    '    transition:all .2s;white-space:nowrap;font-family:inherit;">'
    '    __VOICE_BTN__'
    '  </button>'
    '  <span id="micStatus" style="font-size:11px;color:#777;flex:1;"></span>'
    '</div>'
    '<script>'
    'let _rec=null,_on=false;'
    'function toggleMic(){_on?stopMic():startMic();}'
    'function startMic(){'
    '  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;'
    "  if(!SR){setS('No STT — use Chrome/Edge');return;}"
    "  _rec=new SR();_rec.lang='__STT_LANG__';_rec.continuous=false;_rec.interimResults=true;"
    '  _rec.onstart=()=>{'
    '    _on=true;'
    "    const b=document.getElementById('micBtn');"
    "    b.textContent='Stop';"
    "    b.style.cssText+='background:#4CAF50;color:#000;border-color:#4CAF50;';"
    "    setS('Listening...');"
    '  };'
    '  _rec.onresult=(e)=>{'
    "    let t='';"
    '    for(let i=e.resultIndex;i<e.results.length;i++)t+=e.results[i][0].transcript;'
    "    const preview=t.length>55?t.substring(0,55)+'...':t;"
    "    setS('>> '+preview);"
    '    if(e.results[e.results.length-1].isFinal)submitQ(t);'
    '  };'
    "  _rec.onerror=(e)=>{setS('Error: '+e.error);resetBtn();};"
    '  _rec.onend=resetBtn;'
    '  _rec.start();'
    '}'
    'function stopMic(){if(_rec)_rec.stop();}'
    'function resetBtn(){'
    '  _on=false;'
    "  const b=document.getElementById('micBtn');"
    "  b.textContent='Voice';"
    "  b.style.background='#252836';b.style.color='#ccc';b.style.borderColor='#3a3d4e';"
    '}'
    "function setS(s){const el=document.getElementById('micStatus');if(el)el.textContent=s;}"
    'function submitQ(text){'
    '  const ta=window.parent.document.querySelector('
    "    'textarea[data-testid=\"stChatInputTextArea\"]');"
    "  if(!ta){setS('Failed: '+text);return;}"
    '  const setter=Object.getOwnPropertyDescriptor('
    "    window.parent.HTMLTextAreaElement.prototype,'value').set;"
    '  setter.call(ta,text);'
    "  ta.dispatchEvent(new Event('input',{bubbles:true}));"
    '  setTimeout(()=>{'
    "    ta.dispatchEvent(new KeyboardEvent('keydown',"
    "      {key:'Enter',keyCode:13,bubbles:true,cancelable:true}));"
    "    setS('Submitted!');"
    "    setTimeout(()=>setS(''),2500);"
    '  },130);'
    '}'
    '</script>'
)

def _voice_html(lang: str = "en") -> str:
    stt_lang  = "it-IT" if lang == "it" else "en-US"
    voice_btn = t("ask_voice_btn")
    return _VOICE_HTML_TMPL.replace("__STT_LANG__", stt_lang).replace("__VOICE_BTN__", voice_btn)





# ── Page layout ────────────────────────────────────────────────────────────────
_lang = st.session_state.get("lang", "en")
left_col, right_col = st.columns([1, 3])

# Left panel ────────────────────────────────────────────────────────────────────
with left_col:
    avatar_ph = st.empty()
    with avatar_ph:
        components.html(_avatar_html(st.session_state.avatar_state), height=215)

    tts_on = st.toggle(t("ask_tts_toggle"), value=st.session_state.tts_enabled)
    st.session_state.tts_enabled = tts_on

    if tts_on:
        components.html(
            """
            <button onclick="window.parent.speechSynthesis.cancel()" style="
                width:100%;padding:7px 0;margin-top:4px;
                background:#3a1a1a;border:1px solid #8B2222;border-radius:6px;
                color:#ff6b6b;font-size:13px;font-weight:600;cursor:pointer;
                font-family:inherit;transition:background .2s;"
                onmouseover="this.style.background='#5a2020'"
                onmouseout="this.style.background='#3a1a1a'">
                ⏹ Stop Speaking
            </button>
            """,
            height=46,
        )

    st.divider()
    st.caption(t("ask_suggested"))
    for _qk in SUGGESTED_KEYS:
        _q = t(_qk)
        if st.button(_q, key=f"sugg_{_qk}", use_container_width=True):
            st.session_state.prefill = _q

    st.divider()
    if st.button(t("ask_btn_clear"), use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.avatar_state = "idle"
        st.rerun()

    st.divider()
    st.caption(t("ask_kb_docs", n=len(kb._docs)))
    if _get_api_key():
        st.success(t("ask_api_active"), icon="🟢")
    elif _HAS_SEMANTIC:
        st.info(t("ask_offline_sem"), icon="🔵")
    else:
        st.info(t("ask_offline_tfidf"), icon="📚")

# Right panel ───────────────────────────────────────────────────────────────────
with right_col:
    st.markdown(f"### {t('ask_ai_hdr')}")
    st.caption(t("ask_voice_caption"))

    components.html(_voice_html(_lang), height=50)

    # Welcome message when chat is empty
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown(t("ask_welcome"))

    # Chat history
    for _msg in st.session_state.chat_history:
        with st.chat_message(_msg["role"]):
            st.markdown(_msg["content"])
            if _msg["role"] == "assistant" and _msg.get("sources"):
                with st.expander(t("ask_sources"), expanded=False):
                    for _score, _doc in _msg["sources"]:
                        st.markdown(f"**{_doc['title']}** — relevance: {_score:.3f}")

    # TTS playback — inject a 0-height iframe that speaks the latest answer
    if st.session_state.tts_enabled and st.session_state.answer_tts:
        components.html(
            _tts_html(st.session_state.answer_tts, st.session_state.answer_counter, _lang),
            height=0,
        )

    # ── Input ──────────────────────────────────────────────────────────────────
    _prefill  = st.session_state.pop("prefill", "")
    _question = st.chat_input(t("ask_placeholder")) or _prefill

    if _question:
        # Flip avatar to thinking before processing
        with avatar_ph:
            components.html(_avatar_html("thinking"), height=215)

        with st.chat_message("user"):
            st.markdown(_question)
        st.session_state.chat_history.append({"role": "user", "content": _question})

        _results = kb.retrieve(_question, top_k=3)

        if _get_api_key():
            _ctx = [f"### {d['title']}\n{d['content']}" for _, d in _results]
            with st.spinner(t("ask_spinner")):
                _answer = generate_answer(_question, _ctx)
            if not _answer:
                _answer = compose_local_answer(_question, _results, kb)
        else:
            _answer = compose_local_answer(_question, _results, kb)

        with st.chat_message("assistant"):
            st.markdown(_answer)
            with st.expander(t("ask_sources"), expanded=False):
                for _score, _doc in _results:
                    st.markdown(f"**{_doc['title']}** — relevance: {_score:.3f}")
                    st.caption(_doc["content"][:300] + "…")

        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": _answer,
            "sources": _results,
        })

        # Prepare TTS text (strip markdown, cap at 400 chars)
        _clean = re.sub(r"\*\*(.+?)\*\*", r"\1", _answer)
        _clean = re.sub(r"[#*`]", "", _clean).strip()
        st.session_state.answer_tts     = _clean[:400]
        st.session_state.answer_counter += 1
        st.session_state.avatar_state   = "idle"

        st.rerun()
