/* ==========================================================================
   OPTIC console.

   Performance notes, because this is a real-time UI:
   - Chat is a manually-parsed SSE stream over fetch (EventSource is GET-only),
     so text paints at time-to-first-token instead of time-to-full-answer.
   - Camera frames are pushed on an interval and gated server-side; the encode
     runs on an OffscreenCanvas-sized 2D context we allocate once, never per frame.
   - Voice output speaks sentence-by-sentence as tokens arrive, so speech starts
     while the model is still generating.
   - Every DOM write during streaming touches one text node. No innerHTML in the
     token loop.
   ========================================================================== */

'use strict';

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
// ?static=1 renders everything with motion off — for screenshots, printing,
// and deep links where a scroll reveal would leave the target invisible.
const STATIC = new URLSearchParams(location.search).has('static');
const API = '';                       // same origin

const state = {
  ready: false,
  streaming: false,
  camera: false,
  recording: false,
  voice: true,
  vlmCalls: 0,
  framesSent: 0,
  framesGated: 0,
  bootTime: Date.now(),
  lastTtft: null,
};

/* ======================================================================
   Toasts
   ====================================================================== */
function toast(message, kind = '') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  $('#toasts').appendChild(el);
  if (window.gsap && !REDUCED) {
    gsap.fromTo(el, { x: 28, opacity: 0 }, { x: 0, opacity: 1, duration: .32, ease: 'power2.out' });
    gsap.to(el, { x: 28, opacity: 0, duration: .28, delay: 4, ease: 'power2.in',
                  onComplete: () => el.remove() });
  } else {
    setTimeout(() => el.remove(), 4200);
  }
}

/* ======================================================================
   Boot sequence — typed lines, then a live status poll takes over
   ====================================================================== */
const BOOT = [
  ['core.config',      'loading environment'],
  ['memory.embeddings','sentence-transformers · 384d'],
  ['memory.store',     'vector index mounted'],
  ['brain.provider',   'llm stream ready'],
  ['speech.stt',       'whisper-large-v3-turbo'],
  ['speech.tts',       'browser synthesis'],
  ['vision.engine',    'scene-change gate armed'],
  ['agents.tools',     'tool registry online'],
];

async function runBoot() {
  const host = $('#boot-lines');
  const clock = $('#boot-clock');
  const tick = () => clock.textContent = new Date().toTimeString().slice(0, 8);
  tick(); setInterval(tick, 1000);

  for (const [mod, msg] of BOOT) {
    const line = document.createElement('div');
    line.className = 'boot-line';
    line.innerHTML = `<span class="k">▸</span> ${mod.padEnd(19, ' ')} ${msg}`;
    host.appendChild(line);
    if (window.gsap && !REDUCED) {
      gsap.from(line, { opacity: 0, x: -8, duration: .22, ease: 'power2.out' });
    }
    await new Promise(r => setTimeout(r, REDUCED ? 0 : 95));
  }
  const caret = document.createElement('div');
  caret.className = 'boot-line';
  caret.innerHTML = `<span class="k">▸</span> ready <span class="caret"></span>`;
  host.appendChild(caret);
}

/* ======================================================================
   Status polling
   ====================================================================== */
function setDot(el, cls) { el.className = `dot ${cls}`; }

async function pollStatus() {
  try {
    const r = await fetch(`${API}/api/status`);
    if (!r.ok) throw new Error(r.status);
    const s = await r.json();
    state.ready = true;

    setDot($('#s-link'), 'ok live');
    $('#s-link-t').textContent = 'online';
    $('#s-model').textContent = s.model || '—';
    $('#s-stt').textContent   = s.stt || '—';
    $('#s-vis').textContent   = s.vision_online ? 'on' : 'off';
    $('#s-mem').textContent   = s.memory_facts ?? 0;

    $('#m-turns').textContent = s.turns ?? 0;
    $('#m-mem').textContent   = s.memory_facts ?? 0;
    $('#mem-count').textContent = s.memory_facts ?? 0;

    $('#foot-stack').textContent =
      `${s.provider}/${s.model} · ${s.stt} · ${s.embeddings}`;

    // subsystem rows
    $('#subsystems').innerHTML = [
      ['provider',  s.provider,                        'ok'],
      ['model',     s.model,                           'ok'],
      ['stt',       s.stt,                             'ok'],
      ['tts',       'browser · speechSynthesis',       'ok'],
      ['embeddings',`${s.embeddings} ${s.embedding_dim}d`, s.embeddings === 'hash' ? 'warn' : 'ok'],
      ['vision',    s.vision_model || 'offline',       s.vision_online ? 'ok' : 'off'],
      ['tools',     (s.tools || []).length + ' registered', 'ok'],
      ['episodic',  `${s.episodic} events`,            'ok'],
    ].map(([l, v, c]) =>
      `<div class="row"><span class="l">${l}</span><span class="v ${c}">${esc(String(v))}</span></div>`
    ).join('');

    // latency rows
    const lat = s.latency || {};
    const keys = Object.keys(lat);
    $('#latency').innerHTML = keys.length
      ? keys.map(k => {
          const v = lat[k];
          return `<div class="row"><span class="l">${esc(k)}</span>` +
                 `<span class="v">${v.p50_ms.toFixed(0)} / ${v.p95_ms.toFixed(0)}` +
                 `<span style="color:var(--ink-faint)"> · n${v.count}</span></span></div>`;
        }).join('')
      : `<div class="row"><span class="l">—</span><span class="v off">no samples</span></div>`;

    if (s.perception) renderPerception(s.perception);
  } catch (e) {
    setDot($('#s-link'), 'err');
    $('#s-link-t').textContent = 'offline';
  }
}

function esc(s) {
  return s.replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ======================================================================
   Chat — SSE over fetch
   ====================================================================== */
function addMsg(who, text = '', cls = '') {
  $('#chat-empty')?.remove();
  const wrap = document.createElement('div');
  wrap.className = `msg ${cls}`;
  const label = who === 'you' ? 'you' : who === 'sys' ? 'system' : 'optic';
  wrap.innerHTML = `<div class="who">${label}</div><div class="bubble"></div>`;
  const bubble = $('.bubble', wrap);
  bubble.textContent = text;
  $('#chat').appendChild(wrap);
  scrollChat();
  if (window.gsap && !REDUCED) {
    gsap.from(wrap, { opacity: 0, y: 10, duration: .3, ease: 'power2.out' });
  }
  return { wrap, bubble };
}

function scrollChat() {
  const c = $('#chat');
  c.scrollTop = c.scrollHeight;
}

async function send(text) {
  if (!text || state.streaming) return;
  state.streaming = true;
  $('#input').value = '';
  addMsg('you', text, 'user');

  const { wrap, bubble } = addMsg('optic', '');
  bubble.innerHTML = `<span class="thinking"><i></i><i></i><i></i></span>`;

  const speaker = makeSpeaker();
  let acc = '';
  let firstToken = true;

  try {
    const res = await fetch(`${API}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, use_scene: state.camera }),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      let idx;
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const ev = /^event: (.+)$/m.exec(frame)?.[1];
        const raw = /^data: (.+)$/m.exec(frame)?.[1];
        if (!ev || !raw) continue;
        const data = JSON.parse(raw);

        if (ev === 'start' && data.scene_used) {
          const m = document.createElement('div');
          m.className = 'meta';
          m.innerHTML = `<b>grounded</b> in current view`;
          wrap.appendChild(m);
        } else if (ev === 'ttft') {
          state.lastTtft = data.ms;
          $('#s-ttft').textContent = `${data.ms}ms`;
          $('#m-ttft').innerHTML = `${data.ms}<small>ms</small>`;
        } else if (ev === 'token') {
          if (firstToken) { bubble.textContent = ''; firstToken = false; }
          acc += data.t;
          bubble.textContent = acc;
          speaker.feed(data.t);
          scrollChat();
        } else if (ev === 'error') {
          bubble.textContent = `⚠ ${data.message}`;
          toast(data.message, 'err');
        } else if (ev === 'done') {
          speaker.flush();
          const m = document.createElement('div');
          m.className = 'meta';
          m.innerHTML = `<b>${data.ttft_ms ?? '—'}ms</b> first token · ` +
                        `<b>${data.total_ms}ms</b> total`;
          wrap.appendChild(m);
          $('#m-mem').textContent = data.memory_facts;
          $('#s-mem').textContent = data.memory_facts;
          scrollChat();
        }
      }
    }
  } catch (e) {
    bubble.textContent = `⚠ ${e.message}`;
    toast(`Request failed: ${e.message}`, 'err');
  } finally {
    state.streaming = false;
    pollStatus();
    loadMemory();
  }
}

/* ---------- voice out: speak whole sentences as they complete ---------- */
function makeSpeaker() {
  const synth = window.speechSynthesis;
  if (!state.voice || !synth) return { feed() {}, flush() {} };
  let pending = '';
  const say = (t) => {
    t = t.trim();
    if (!t) return;
    const u = new SpeechSynthesisUtterance(t);
    u.rate = 1.05; u.pitch = 1.0;
    const v = synth.getVoices().find(v => /en[-_]/i.test(v.lang) && /female|samantha|zira|google/i.test(v.name))
           || synth.getVoices().find(v => /en[-_]/i.test(v.lang));
    if (v) u.voice = v;
    synth.speak(u);
  };
  return {
    feed(chunk) {
      pending += chunk;
      // Flush on sentence boundaries so speech starts before generation ends.
      const m = pending.match(/^([\s\S]+?[.!?…]+[\s"')\]]*)/);
      if (m) { say(m[1]); pending = pending.slice(m[0].length); }
    },
    flush() { say(pending); pending = ''; },
  };
}

function stopSpeaking() { window.speechSynthesis?.cancel(); }

/* ======================================================================
   Camera — push frames, let the server gate them
   ====================================================================== */
let camStream = null, camTimer = null, canvas = null, ctx2d = null;

async function toggleCamera() {
  if (state.camera) return stopCamera();
  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    const v = $('#cam');
    v.srcObject = camStream;
    await v.play();

    state.camera = true;
    $('#cam-shell').classList.remove('off');
    $('#cam-toggle').textContent = 'Disable camera';
    $('#cam-look').disabled = false;
    setDot($('#cam-dot'), 'ok live');
    $('#hud-tag').textContent = 'TRACKING';
    $('#cam-res').textContent = `${v.videoWidth}×${v.videoHeight}`;

    // One canvas for the whole session — allocating per frame is what makes
    // naive versions of this stutter.
    canvas = document.createElement('canvas');
    canvas.width = 640; canvas.height = 480;
    ctx2d = canvas.getContext('2d', { alpha: false, willReadFrequently: false });

    camTimer = setInterval(() => pushFrame(false), 1800);
    pushFrame(true);
    toast('Camera online — frames are scene-gated before analysis');
  } catch (e) {
    toast(`Camera denied: ${e.message}`, 'warn');
  }
}

function stopCamera() {
  clearInterval(camTimer); camTimer = null;
  camStream?.getTracks().forEach(t => t.stop());
  camStream = null;
  state.camera = false;
  $('#cam').srcObject = null;
  $('#cam-shell').classList.add('off', '');
  $('#cam-shell').classList.remove('scanning');
  $('#cam-toggle').textContent = 'Enable camera';
  $('#cam-look').disabled = true;
  setDot($('#cam-dot'), '');
  $('#hud-tag').textContent = 'STANDBY';
  $('#cam-res').textContent = '—';
}

async function pushFrame(force) {
  if (!state.camera || !ctx2d) return;
  const v = $('#cam');
  if (!v.videoWidth) return;

  ctx2d.drawImage(v, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.72));
  if (!blob) return;

  const fd = new FormData();
  fd.append('frame', blob, 'frame.jpg');
  fd.append('force', force ? 'true' : 'false');

  state.framesSent++;
  try {
    const r = await fetch(`${API}/api/vision/frame`, { method: 'POST', body: fd });
    const d = await r.json();
    if (d.accepted) {
      state.vlmCalls++;
      $('#m-vlm').textContent = state.vlmCalls;
      $('#cam-shell').classList.add('scanning');
      $('#hud-tag').textContent = 'ANALYSING';
      setDot($('#perc-dot'), 'warn live');
      pollPerception();
    } else {
      if (d.reason === 'no change') state.framesGated++;
    }
    const pct = state.framesSent ? Math.round(state.framesGated / state.framesSent * 100) : 0;
    $('#m-saved').innerHTML = `${pct}<small>%</small>`;
    $('#hud-fps').textContent = `${state.vlmCalls} vlm · ${state.framesGated} gated`;
  } catch { /* transient — next tick retries */ }
}

let percTimer = null;
function pollPerception() {
  clearInterval(percTimer);
  percTimer = setInterval(async () => {
    try {
      const d = await (await fetch(`${API}/api/vision/latest`)).json();
      if (!d.busy) {
        clearInterval(percTimer);
        $('#cam-shell').classList.remove('scanning');
        $('#hud-tag').textContent = state.camera ? 'TRACKING' : 'STANDBY';
        setDot($('#perc-dot'), 'ok');
        if (d.perception) renderPerception(d.perception);
      }
    } catch { clearInterval(percTimer); }
  }, 700);
}

function renderPerception(p) {
  $('#perc-age').textContent = `${p.age_s}s ago`;
  $('#perception').innerHTML =
    `<span class="age">[${p.age_s}s ago]</span> ${esc(p.summary)}`;
}

/* ======================================================================
   Microphone — MediaRecorder -> /api/stt
   ====================================================================== */
let recorder = null, chunks = [], micStream = null, audioCtx = null, rafId = null;

async function toggleMic() {
  if (state.recording) return stopMic();
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    stopSpeaking();                              // barge-in: cut playback

    recorder = new MediaRecorder(micStream);
    chunks = [];
    recorder.ondataavailable = e => e.data.size && chunks.push(e.data);
    recorder.onstop = onMicStop;
    recorder.start();

    state.recording = true;
    $('#btn-mic').classList.add('rec');
    $('#level').hidden = false;
    startMeter();
  } catch (e) {
    toast(`Microphone denied: ${e.message}`, 'warn');
  }
}

function stopMic() {
  if (!state.recording) return;
  state.recording = false;
  $('#btn-mic').classList.remove('rec');
  $('#level').hidden = true;
  cancelAnimationFrame(rafId);
  audioCtx?.close(); audioCtx = null;
  try { recorder?.stop(); } catch {}
  micStream?.getTracks().forEach(t => t.stop());
  micStream = null;
}

async function onMicStop() {
  const blob = new Blob(chunks, { type: 'audio/webm' });
  if (blob.size < 1200) return;                  // click, not speech
  const fd = new FormData();
  fd.append('audio', blob, 'clip.webm');
  const holder = addMsg('sys', 'transcribing…', 'sys');
  try {
    const d = await (await fetch(`${API}/api/stt`, { method: 'POST', body: fd })).json();
    holder.wrap.remove();
    if (d.text?.trim()) send(d.text.trim());
    else toast('Nothing recognised — try again', 'warn');
  } catch (e) {
    holder.bubble.textContent = `⚠ transcription failed`;
    toast(e.message, 'err');
  }
}

function startMeter() {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const src = audioCtx.createMediaStreamSource(micStream);
  const an = audioCtx.createAnalyser();
  an.fftSize = 64;
  src.connect(an);
  const bins = new Uint8Array(an.frequencyBinCount);
  const bars = $$('#level i');

  const draw = () => {
    an.getByteFrequencyData(bins);
    for (let i = 0; i < bars.length; i++) {
      const v = bins[i * 2] || 0;
      bars[i].style.height = `${3 + (v / 255) * 22}px`;
    }
    rafId = requestAnimationFrame(draw);
  };
  draw();
}

/* ======================================================================
   Memory panel
   ====================================================================== */
async function loadMemory() {
  try {
    const d = await (await fetch(`${API}/api/memory?limit=60`)).json();
    $('#mem-count').textContent = d.count;
    $('#mem-all').innerHTML = d.records.length
      ? d.records.map(r => `
          <div class="mem-item" data-id="${r.id}">
            <div class="grow">
              <div>${esc(r.text)}</div>
              <div class="kind">${r.kind} · ${fmtAge(r.age_s)}</div>
            </div>
            <button class="x" title="Forget">×</button>
          </div>`).join('')
      : `<div class="kind" style="color:var(--ink-faint)">no durable memories yet</div>`;

    $('#episodic').innerHTML = d.episodic.length
      ? d.episodic.map(e =>
          `<div><span class="age">[${e.age_s}s]</span> <b style="color:var(--cyan-2);font-weight:500">${e.kind}</b> ${esc(e.text.slice(0, 120))}</div>`
        ).join('')
      : '<em>empty</em>';
  } catch {}
}

function fmtAge(s) {
  if (s < 90) return `${s}s ago`;
  if (s < 5400) return `${Math.floor(s / 60)}m ago`;
  if (s < 172800) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

async function probeMemory() {
  const q = $('#mem-q').value.trim();
  if (!q) return;
  try {
    const d = await (await fetch(`${API}/api/memory/search`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q }),
    })).json();

    const gate = $('#mem-gate');
    gate.className = `gate ${d.gate_open ? 'open' : 'closed'}`;
    gate.innerHTML = `<i class="dot ${d.gate_open ? 'ok' : 'warn'}"></i><span>` +
      (d.gate_open
        ? `gate open — memory searched · threshold ${d.threshold}`
        : `gate closed — general knowledge, no memory injected`) + `</span>`;

    $('#mem-hits').innerHTML = d.hits.length
      ? d.hits.map(h => `
          <div class="mem-item ${h.shown ? 'shown' : 'filtered'}">
            <span class="score">${h.score.toFixed(3)}</span>
            <div class="grow"><div>${esc(h.text)}</div>
              <div class="kind">${h.shown ? 'passed to model' : 'below threshold'}</div>
            </div>
          </div>`).join('')
      : `<div class="kind" style="color:var(--ink-faint)">no vectors in store</div>`;

    const ctx = $('#mem-ctx');
    ctx.hidden = false;
    ctx.textContent = d.context || '(nothing — no memory injected into the prompt)';

    if (window.gsap && !REDUCED) {
      gsap.from('#mem-hits .mem-item', { opacity: 0, x: -10, duration: .28,
                                         stagger: .04, ease: 'power2.out' });
    }
  } catch (e) { toast(e.message, 'err'); }
}

/* ======================================================================
   Motion — Lenis smooth scroll driving GSAP ScrollTrigger
   ====================================================================== */
function revealAll() { $$('.rv').forEach(e => e.classList.remove('rv')); }

function initMotion() {
  // `.rv` starts at opacity:0 and GSAP animates it in. If GSAP is missing or
  // slow, that would leave the page blank — so anything still hidden after a
  // beat gets shown unconditionally.
  setTimeout(() => { if (!window.__motionReady) revealAll(); }, 2500);

  if (REDUCED || STATIC || !window.gsap) { revealAll(); return; }
  window.__motionReady = true;

  gsap.registerPlugin(ScrollTrigger);

  let lenis = null;
  if (window.Lenis) {
    lenis = new Lenis({
      duration: 1.05,
      easing: t => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
      touchMultiplier: 1.6,
    });
    // One rAF loop drives Lenis; ScrollTrigger updates from its scroll event.
    lenis.on('scroll', ScrollTrigger.update);
    gsap.ticker.add(t => lenis.raf(t * 1000));
    gsap.ticker.lagSmoothing(0);
    window.__lenis = lenis;
  }

  // Hero: mask-reveal each headline line, then the supporting content.
  const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });
  tl.from('.hero h1 .line > span', { yPercent: 112, duration: .9, stagger: .1 })
    .from('.hero .eyebrow',  { opacity: 0, y: 12, duration: .5 }, 0.1)
    .from('.hero .lede',     { opacity: 0, y: 16, duration: .6 }, 0.45)
    .from('.hero-cta .btn',  { opacity: 0, y: 14, duration: .45, stagger: .07 }, 0.6)
    .from('.boot',           { opacity: 0, y: 20, duration: .7 }, 0.25)
    .from('.scroll-cue',     { opacity: 0, duration: .5 }, 0.9);

  // Section reveals.
  $$('.rv').forEach(el => {
    gsap.to(el, {
      opacity: 1, y: 0, duration: .75, ease: 'power3.out',
      scrollTrigger: { trigger: el, start: 'top 86%', once: true },
    });
  });

  // Pipeline: stagger the nodes, then draw each progress bar as it enters.
  gsap.to('.pipe .node', {
    opacity: 1, y: 0, duration: .6, stagger: .1, ease: 'power3.out',
    scrollTrigger: { trigger: '.pipe', start: 'top 78%', once: true },
  });
  $$('.pipe .node').forEach((n, i) => {
    gsap.to($('.bar', n), {
      scaleX: 1, duration: 1.1, ease: 'power2.out', delay: i * 0.08,
      scrollTrigger: { trigger: n, start: 'top 82%', once: true },
    });
  });

  // Metric tiles count up when they scroll into view.
  ScrollTrigger.create({
    trigger: '.tiles', start: 'top 84%', once: true,
    onEnter: () => gsap.from('.tile', { opacity: 0, y: 18, duration: .55,
                                        stagger: .06, ease: 'power3.out' }),
  });

  // Parallax the ambient field very slightly against scroll.
  gsap.to('body', {
    '--parallax': 1,
    scrollTrigger: { trigger: 'body', start: 'top top', end: 'bottom bottom', scrub: .6 },
  });

  // Smooth in-page nav through Lenis so the easing matches.
  $$('[data-scroll-to]').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = $(btn.dataset.scrollTo);
      if (!target) return;
      lenis ? lenis.scrollTo(target, { offset: -60 })
            : target.scrollIntoView({ behavior: 'smooth' });
    });
  });
}

/* ======================================================================
   Wiring
   ====================================================================== */
function initEvents() {
  $('#btn-send').addEventListener('click', () => send($('#input').value.trim()));
  $('#input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send($('#input').value.trim()); }
  });

  $('#btn-mic').addEventListener('click', toggleMic);
  $('#cam-toggle').addEventListener('click', toggleCamera);
  $('#cam-look').addEventListener('click', () => {
    pushFrame(true);
    toast('Forcing a fresh look…');
  });

  $('#btn-voice').addEventListener('click', e => {
    state.voice = !state.voice;
    e.target.textContent = state.voice ? 'Voice on' : 'Voice off';
    if (!state.voice) stopSpeaking();
  });

  $('#btn-reset').addEventListener('click', async () => {
    await fetch(`${API}/api/reset`, { method: 'POST' });
    stopSpeaking();
    $('#chat').innerHTML =
      `<div class="chat-empty" id="chat-empty">conversation cleared<br>` +
      `<kbd>durable memory kept</kbd></div>`;
    toast('Conversation history cleared (memory kept)');
    pollStatus();
  });

  $('#mem-go').addEventListener('click', probeMemory);
  $('#mem-q').addEventListener('keydown', e => e.key === 'Enter' && probeMemory());

  // Delete a memory (delegated — the list re-renders constantly).
  $('#mem-all').addEventListener('click', async e => {
    const btn = e.target.closest('.x');
    if (!btn) return;
    const item = btn.closest('.mem-item');
    const id = item.dataset.id;
    await fetch(`${API}/api/memory/${id}`, { method: 'DELETE' });
    if (window.gsap && !REDUCED) {
      gsap.to(item, { opacity: 0, x: -20, height: 0, duration: .25,
                      onComplete: () => { item.remove(); loadMemory(); } });
    } else { item.remove(); loadMemory(); }
    toast(`Forgot memory ${id}`);
  });

  // Space-to-talk when not typing.
  document.addEventListener('keydown', e => {
    if (e.code === 'Space' && e.target.tagName !== 'INPUT' && !e.repeat) {
      e.preventDefault(); toggleMic();
    }
    if (e.key === 'Escape') { stopSpeaking(); stopMic(); }
  });

  // Stop the camera when the tab is hidden — no point burning frames.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && camTimer) { clearInterval(camTimer); camTimer = null; }
    else if (!document.hidden && state.camera && !camTimer) {
      camTimer = setInterval(() => pushFrame(false), 1800);
    }
  });
}

function uptimeTicker() {
  setInterval(() => {
    const s = Math.floor((Date.now() - state.bootTime) / 1000);
    const m = Math.floor(s / 60);
    $('#foot-uptime').textContent = `uptime ${m}m ${String(s % 60).padStart(2, '0')}s`;
  }, 1000);
}

/* ---------- go ---------- */
window.addEventListener('DOMContentLoaded', () => {
  initEvents();
  initMotion();
  runBoot();
  uptimeTicker();
  pollStatus();
  loadMemory();
  setInterval(pollStatus, 5000);
  // Chrome populates voices asynchronously.
  window.speechSynthesis?.getVoices();
});
