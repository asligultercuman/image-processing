# server.py
from flask import Flask, request, send_file, jsonify
import tempfile
import os
import cv2
from objectDetection import ObjectDetector
import utils

app = Flask(__name__)

# Yüklenen dosya boyutu limiti: 500 MB
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


@app.route("/process", methods=["POST"])
def process():
    # ── Giriş doğrulama ──────────────────────────────────────────────────────
    if "video" not in request.files:
        return jsonify({"error": "Video dosyası bulunamadı."}), 400

    video_file = request.files["video"]
    hedef      = request.form.get("hedef")
    kaynak     = request.form.get("kaynak") or None   # boş string → None
    sinif      = request.form.get("sinif")  or None

    if not hedef:
        return jsonify({"error": "Hedef renk belirtilmedi."}), 400

    # ── Geçici dosyalar ───────────────────────────────────────────────────────
    suffix = ".mp4"
    fname  = video_file.filename or ""
    if fname.lower().endswith(".avi"):
        suffix = ".avi"

    tmp_in  = tempfile.NamedTemporaryFile(suffix=suffix,  delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4",  delete=False)
    tmp_in.close()
    tmp_out.close()

    try:
        video_file.save(tmp_in.name)

        # ── Video aç ──────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(tmp_in.name)
        if not cap.isOpened():
            return jsonify({"error": "Video dosyası açılamadı."}), 422

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── VideoWriter ───────────────────────────────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_out.name, fourcc, fps, (width, height))

        # ── Detector ─────────────────────────────────────────────────────────
        detector = ObjectDetector(
            renk_komutu={
                "hedef_sinif": sinif,
                "kaynak_renk": kaynak,
                "hedef_renk":  hedef,
            }
        )

        # ── Frame döngüsü ─────────────────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = utils.apply_clahe(frame)
            boxes   = detector.detect(frame)
            frame   = detector.draw(frame, boxes)
            writer.write(frame)

        cap.release()
        writer.release()

        # ── Çıktıyı gönder ───────────────────────────────────────────────────
        out_name = fname.rsplit(".", 1)[0] + f"_{hedef}_output.mp4" if fname else "output.mp4"
        return send_file(
            tmp_out.name,
            mimetype="video/mp4",
            as_attachment=True,
            download_name=out_name,
        )

    finally:
        # Geçici dosyaları temizle (send_file gönderdikten sonra da çalışır)
        for path in (tmp_in.name,):
            try:
                os.unlink(path)
            except OSError:
                pass
        # tmp_out'u send_file okuduktan sonra sil — arka planda temizle
        # (Windows'ta send_file bitmeden silinemez, Linux'ta sorun yok)


@app.route("/", methods=["GET"])
def index():
    """HTML arayüzünü sunar."""
    html = """<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Renk Editörü</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tabler-icons/2.44.0/tabler-icons.min.css">
    <style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; }

#app {
  display: flex;
  height: 100vh;
  background: white;
}

#sidebar {
  width: 240px;
  min-width: 240px;
  border-right: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
  background: #fafafa;
}

#sidebar-header {
  padding: 14px 16px 10px;
  border-bottom: 1px solid #e0e0e0;
}

#sidebar-header h3 {
  font-size: 13px;
  font-weight: 600;
  color: #333;
  margin-bottom: 2px;
}

#sidebar-header p {
  font-size: 11px;
  color: #999;
}

#video-drop {
  margin: 12px;
  border: 1.5px dashed #ddd;
  border-radius: 8px;
  padding: 18px 12px;
  text-align: center;
  cursor: pointer;
  transition: all 0.15s;
  position: relative;
  background: white;
}

#video-drop:hover, #video-drop.drag-over {
  background: #e3f2fd;
  border-color: #2196f3;
}

#video-drop input[type=file] {
  position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%;
}

#video-drop i { font-size: 24px; color: #999; display: block; margin-bottom: 6px; }
#video-drop .drop-label { font-size: 12px; color: #666; font-weight: 500; }
#video-drop .drop-sub { font-size: 11px; color: #999; margin-top: 3px; }

#video-preview-wrap {
  margin: 0 12px 12px;
  display: none;
}

#video-preview-wrap video {
  width: 100%;
  border-radius: 8px;
  border: 1px solid #ddd;
  display: block;
}

#video-name {
  font-size: 11px;
  color: #666;
  margin-top: 6px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

#active-cmd-label {
  padding: 0 12px 4px;
  font-size: 10px;
  font-weight: 600;
  color: #999;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

#active-cmd-box {
  margin: 0 12px 8px;
  padding: 8px 10px;
  background: white;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 12px;
  color: #666;
  min-height: 38px;
  line-height: 1.6;
}

#active-cmd-box span.tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 500;
  margin: 2px 3px 0 0;
  color: #fff;
}

.tag-src { background: #2196f3; }
.tag-dst { background: #4caf50; }
.tag-cls { background: #ff9800; }

#run-btn {
  margin: auto 12px 14px;
  width: calc(100% - 24px);
  padding: 10px;
  background: white;
  border: 1px solid #2196f3;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  color: #2196f3;
  cursor: pointer;
  display: flex; align-items: center; justify-content: center; gap: 6px;
  transition: all 0.12s;
}

#run-btn:hover:not(:disabled) { background: #e3f2fd; }
#run-btn:disabled { opacity: 0.4; cursor: not-allowed; }

#chat-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

#chat-header {
  padding: 12px 16px;
  border-bottom: 1px solid #e0e0e0;
  display: flex; align-items: center; gap: 8px;
}

#chat-header i { font-size: 18px; color: #2196f3; }
#chat-header span { font-size: 14px; font-weight: 600; color: #333; }

#chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.msg {
  display: flex;
  gap: 10px;
  max-width: 88%;
  animation: fadein 0.2s ease;
}

@keyframes fadein { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; } }

.msg.user { align-self: flex-end; flex-direction: row-reverse; }

.msg-avatar {
  width: 28px; height: 28px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 600;
  flex-shrink: 0;
  margin-top: 2px;
  color: #fff;
}

.msg.user .msg-avatar { background: #2196f3; }
.msg.bot .msg-avatar { background: #f0f0f0; border: 1px solid #ddd; color: #666; }

.msg-bubble {
  padding: 10px 12px;
  border-radius: 14px;
  font-size: 13.5px;
  line-height: 1.55;
  color: #333;
  background: #f0f0f0;
  border: 1px solid #ddd;
}

.msg.user .msg-bubble {
  background: #2196f3;
  border: none;
  color: #fff;
}

.msg-bubble .chip {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 8px; border-radius: 12px; font-size: 11.5px;
  margin: 3px 2px 0 0;
  color: #fff;
}

.chip-ok { background: #4caf50; }
.chip-warn { background: #ff9800; }
.chip-err { background: #f44336; }

.download-row {
  margin-top: 8px;
  display: flex; gap: 8px; flex-wrap: wrap;
}

.dl-btn {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 6px 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 12px; font-weight: 500;
  color: #2196f3;
  background: #fff;
  cursor: pointer; text-decoration: none;
  transition: all 0.12s;
}

.dl-btn:hover { background: #e3f2fd; border-color: #2196f3; }

.progress-bar-wrap {
  margin-top: 8px;
  height: 4px;
  background: #ddd;
  border-radius: 99px;
  overflow: hidden;
  width: 100%;
}

.progress-bar {
  height: 100%;
  background: #2196f3;
  transition: width 0.3s ease;
}

#input-area {
  padding: 12px 16px;
  border-top: 1px solid #e0e0e0;
  display: flex; gap: 8px; align-items: flex-end;
}

#cmd-input {
  flex: 1;
  resize: none;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 9px 12px;
  font-size: 13.5px;
  font-family: inherit;
  color: #333;
  background: #fff;
  line-height: 1.5;
  min-height: 38px;
  max-height: 100px;
  overflow-y: auto;
}

#cmd-input:focus { outline: none; border-color: #2196f3; box-shadow: 0 0 0 2px #e3f2fd; }
#cmd-input::placeholder { color: #999; }

#send-btn {
  width: 36px; height: 36px;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  color: #2196f3;
  flex-shrink: 0;
  transition: all 0.12s;
  font-size: 0;
}

#send-btn:hover { background: #e3f2fd; border-color: #2196f3; }
#send-btn i { font-size: 16px; }

.hint-pills {
  display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 0;
}

.hint-pill {
  padding: 4px 10px;
  border: 1px solid #ddd;
  border-radius: 12px;
  font-size: 11.5px;
  color: #666;
  cursor: pointer;
  transition: all 0.12s;
  background: #fff;
}

.hint-pill:hover {
  background: #e3f2fd;
  border-color: #2196f3;
  color: #2196f3;
}
    </style>
</head>
<body>

<div id="app">
  <div id="sidebar">
    <div id="sidebar-header">
      <h3>📹 Video dosyası</h3>
      <p>mp4 veya avi yükleyin</p>
    </div>

    <div id="video-drop">
      <input type="file" id="file-input" accept=".mp4,.avi,video/mp4,video/x-msvideo">
      <i class="ti ti-upload"></i>
      <div class="drop-label">Dosya seç veya sürükle</div>
      <div class="drop-sub">mp4 · avi</div>
    </div>

    <div id="video-preview-wrap">
      <video id="video-preview" controls muted></video>
      <div id="video-name"></div>
    </div>

    <div id="active-cmd-label">Aktif komut</div>
    <div id="active-cmd-box">
      <span style="color:#999;font-size:12px;">Henüz komut girilmedi</span>
    </div>

    <button id="run-btn" disabled>
      <i class="ti ti-player-play"></i>
      İşlemi başlat
    </button>
  </div>

  <div id="chat-panel">
    <div id="chat-header">
      <i class="ti ti-wand"></i>
      <span>Video Renk Editörü</span>
    </div>

    <div id="chat-messages">
      <div class="msg bot">
        <div class="msg-avatar"><i class="ti ti-wand"></i></div>
        <div class="msg-bubble">
          Merhaba! Bir video yükleyip renk değiştirme komutu girebilirsin.
          <div class="hint-pills">
            <span class="hint-pill" onclick="fillHint(this)">mavi gömleği yeşile çevir</span>
            <span class="hint-pill" onclick="fillHint(this)">arabayı siyah yap</span>
            <span class="hint-pill" onclick="fillHint(this)">sarı arabayı kırmızıya boya</span>
          </div>
        </div>
      </div>
    </div>

    <div id="input-area">
      <textarea id="cmd-input" rows="1" placeholder="Komut girin… (ör. mavi gömleği yeşile çevir)"></textarea>
      <button id="send-btn" aria-label="Gönder" onclick="sendCommand()">
        <i class="ti ti-send"></i>
      </button>
    </div>
  </div>
</div>

<script>
const fileInput = document.getElementById('file-input');
const videoDrop = document.getElementById('video-drop');
const videoPreview = document.getElementById('video-preview');
const videoPreviewWrap = document.getElementById('video-preview-wrap');
const videoName = document.getElementById('video-name');
const activeCmdBox = document.getElementById('active-cmd-box');
const runBtn = document.getElementById('run-btn');
const cmdInput = document.getElementById('cmd-input');
const chatMessages = document.getElementById('chat-messages');

let currentFile = null;
let parsedCmd = null;

const COLOR_KEYWORDS = {
  red: ['kirmizi','kizil','red'],
  orange: ['turuncu','orange'],
  blue: ['mavi','lacivert','blue'],
  green: ['yesil','green'],
  yellow: ['sari','yellow'],
  cyan: ['turkuaz','cyan','acik mavi'],
  purple: ['mor','purple'],
  pink: ['pembe','pink'],
  white: ['beyaz','white'],
  black: ['siyah','black'],
  gray: ['gri','gray','grey']
};

const CLASS_KEYWORDS = {
  person: ['insan','kisi','adam','kadin','person','gomlek','kisinin','adamin'],
  car: ['araba','otomobil','car','arabanin','arabanın'],
  truck: ['kamyon','tir','truck'],
  bus: ['otobus','bus'],
  motorcycle: ['motor','motosiklet'],
  bicycle: ['bisiklet','bicycle']
};

function normalize(t) {
  t = (t||'').toLowerCase().trim();
  const map = {'ı':'i','ğ':'g','ü':'u','ş':'s','ö':'o','ç':'c','İ':'i','Ğ':'g','Ü':'u','Ş':'s','Ö':'o','Ç':'c'};
  return t.replace(/[ığüşöçİĞÜŞÖÇ]/g, m => map[m]||m).replace(/[^\\w\\s]/g,' ').replace(/\\s+/g,' ').trim();
}

function extractColors(t) {
  const found = [];
  for (const [color, kws] of Object.entries(COLOR_KEYWORDS)) {
    for (const kw of kws) {
      const idx = t.indexOf(kw);
      if (idx !== -1) { found.push([idx, color]); break; }
    }
  }
  return found.sort((a,b)=>a[0]-b[0]).map(x=>x[1]);
}

function extractClass(t) {
  for (const [cls, kws] of Object.entries(CLASS_KEYWORDS)) {
    if (kws.some(k => t.includes(k))) return cls;
  }
  return null;
}

function parseCommand(raw) {
  const t = normalize(raw);
  const colors = extractColors(t);
  const cls = extractClass(t);
  const CANCEL = ['renk degistirmeyi kapat','renk iptal','rengi geri al','renk kapat'];
  const TRIGGERS = ['rengini','renge cevir','rene boya','rengini degistir','boyasini','boya','cevir'];

  if (CANCEL.some(c => t.includes(c))) return { action: 'renk_iptal' };

  const hasTrigger = TRIGGERS.some(k => t.includes(k));
  const hasYap = t.includes('yap');

  if (colors.length >= 2) {
    return { action: 'renk_degistir', kaynak: colors[0], hedef: colors[1], sinif: cls };
  }
  if (colors.length === 1 && (hasTrigger || hasYap)) {
    return { action: 'renk_degistir', kaynak: null, hedef: colors[0], sinif: cls };
  }
  if (hasTrigger && colors.length >= 1) {
    return { action: 'renk_degistir', kaynak: colors[0], hedef: null, sinif: cls };
  }
  return { action: 'unknown' };
}

function updateActiveCmdBox(p) {
  if (!p || p.action === 'unknown') {
    activeCmdBox.innerHTML = '<span style="color:#999;font-size:12px;">Komut anlaşılamadı</span>';
    return;
  }
  if (p.action === 'renk_iptal') {
    activeCmdBox.innerHTML = '<span style="color:#999;font-size:12px;">Renk değiştirme kapatılacak</span>';
    return;
  }
  let html = '';
  if (p.kaynak) html += `<span class="tag tag-src">kaynak: ${p.kaynak}</span>`;
  if (p.hedef)  html += `<span class="tag tag-dst">hedef: ${p.hedef}</span>`;
  if (p.sinif)  html += `<span class="tag tag-cls">sınıf: ${p.sinif}</span>`;
  if (!p.kaynak && p.hedef) html += `<span class="tag tag-src" style="opacity:0.6;">kaynak: otomatik</span>`;
  activeCmdBox.innerHTML = html || '<span style="color:#999;font-size:12px;">—</span>';
}

function addMessage(role, html) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  if (role === 'user') avatar.textContent = 'S';
  else avatar.innerHTML = '<i class="ti ti-wand"></i>';
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.innerHTML = html;
  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  chatMessages.appendChild(wrap);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return bubble;
}

function fillHint(el) {
  cmdInput.value = el.textContent;
  cmdInput.focus();
  autoResize();
}

function autoResize() {
  cmdInput.style.height = 'auto';
  cmdInput.style.height = Math.min(cmdInput.scrollHeight, 100) + 'px';
}

cmdInput.addEventListener('input', autoResize);
cmdInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendCommand(); }
});

function sendCommand() {
  const raw = cmdInput.value.trim();
  if (!raw) return;
  addMessage('user', raw);
  cmdInput.value = '';
  autoResize();

  const p = parseCommand(raw);
  parsedCmd = p;
  updateActiveCmdBox(p);

  if (p.action === 'renk_iptal') {
    addMessage('bot', 'Renk değiştirme komutu temizlendi. Yeni bir komut girebilirsin.');
    updateRunBtn();
    return;
  }

  if (p.action === 'unknown') {
    addMessage('bot', `<span class="chip chip-warn"><i class="ti ti-alert-triangle"></i> Komut anlaşılamadı</span><br>Örnek: <em>"mavi gömleği yeşile çevir"</em>`);
    return;
  }

  let resp = '<span class="chip chip-ok"><i class="ti ti-check"></i> Komut alındı</span><br>';
  if (p.kaynak && p.hedef) resp += `<strong>${p.kaynak}</strong> → <strong>${p.hedef}</strong> renk dönüşümü.`;
  else if (!p.kaynak && p.hedef) resp += `Tüm renkler → <strong>${p.hedef}</strong> olarak değişecek.`;
  if (p.sinif) resp += ` (${p.sinif} sınıfı)`;
  addMessage('bot', resp);
  updateRunBtn();
}

function updateRunBtn() {
  runBtn.disabled = !(currentFile && parsedCmd && parsedCmd.action === 'renk_degistir');
}

fileInput.addEventListener('change', e => {
  const f = e.target.files[0];
  if (f) loadFile(f);
});

videoDrop.addEventListener('dragover', e => { e.preventDefault(); videoDrop.classList.add('drag-over'); });
videoDrop.addEventListener('dragleave', () => videoDrop.classList.remove('drag-over'));
videoDrop.addEventListener('drop', e => {
  e.preventDefault();
  videoDrop.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && (f.name.endsWith('.mp4')||f.name.endsWith('.avi'))) loadFile(f);
  else addMessage('bot', '<span class="chip chip-err"><i class="ti ti-x"></i> Yalnızca mp4 veya avi yüklenebilir</span>');
});

function loadFile(f) {
  currentFile = f;
  const url = URL.createObjectURL(f);
  videoPreview.src = url;
  videoPreviewWrap.style.display = 'block';
  videoName.textContent = f.name;
  videoDrop.style.display = 'none';
  addMessage('bot', `<span class="chip chip-ok"><i class="ti ti-check"></i> Video yüklendi</span><br><strong>${f.name}</strong> hazır.`);
  updateRunBtn();
}

runBtn.addEventListener('click', async () => {
  if (!currentFile || !parsedCmd) return;
  runBtn.disabled = true;

  const bubble = addMessage('bot',
    `<i class="ti ti-loader"></i> İşleniyor… <strong>${currentFile.name}</strong>
     <div class="progress-bar-wrap"><div class="progress-bar" id="pbar" style="width:0%"></div></div>`
  );

  let prog = 0;
  const pbar = bubble.querySelector('#pbar');
  const interval = setInterval(() => {
    prog = Math.min(prog + Math.random() * 8, 88);
    if (pbar) pbar.style.width = prog.toFixed(0) + '%';
  }, 300);

  try {
    const fd = new FormData();
    fd.append('video',  currentFile);
    fd.append('hedef',  parsedCmd.hedef);
    fd.append('kaynak', parsedCmd.kaynak || '');
    fd.append('sinif',  parsedCmd.sinif  || '');

    const res = await fetch('/process', {
      method: 'POST',
      body: fd,
    });

    clearInterval(interval);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || 'Sunucu hatası');
    }

    const blob = await res.blob();
    const outName = currentFile.name.replace(/\\.(mp4|avi)$/i, `_${parsedCmd.hedef}_output.mp4`);
    const url = URL.createObjectURL(blob);

    if (pbar) pbar.style.width = '100%';

    const cmdStr = parsedCmd.kaynak
      ? `${parsedCmd.kaynak} → ${parsedCmd.hedef}`
      : `* → ${parsedCmd.hedef}`;
    const clsStr = parsedCmd.sinif ? ` (${parsedCmd.sinif})` : '';

    bubble.innerHTML = `
      <span class="chip chip-ok"><i class="ti ti-check"></i> Tamamlandı</span><br>
      Komut <strong>${cmdStr}${clsStr}</strong> uygulandı.<br>
      <div class="download-row">
        <a class="dl-btn" href="${url}" download="${outName}">
          <i class="ti ti-download"></i> ${outName}
        </a>
      </div>
    `;

  } catch (err) {
    clearInterval(interval);
    bubble.innerHTML = `
      <span class="chip chip-err"><i class="ti ti-x"></i> Hata</span><br>
      ${err.message}
    `;
  } finally {
    runBtn.disabled = false;
  }
});
</script>

</body>
</html>"""
    return html


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # debug=False — production'da Waitress veya Gunicorn kullan
    app.run(host="0.0.0.0", port=5000, debug=False)