const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.defineLayout({ name: "W", width: 13.33, height: 7.5 });
p.layout = "W";

const NAVY = "1E2761", ICE = "CADCFC", WHITE = "FFFFFF",
      DARK = "222B45", MUTED = "6B7488", ACCENT = "3D5AFE";
const TITLE_FONT = "Georgia", BODY_FONT = "Calibri";

// footer slide number on content slides
function footer(s, n) {
  s.addText([{ text: "KSL-LLM-IoT  ·  IoT Systems Term Project", options: { color: MUTED, fontSize: 9 } }],
    { x: 0.5, y: 7.05, w: 9, h: 0.3, align: "left", fontFace: BODY_FONT });
  s.addText(`${n}`, { x: 12.4, y: 7.05, w: 0.5, h: 0.3, align: "right", color: MUTED, fontSize: 9, fontFace: BODY_FONT });
}

// content slide: title + bullets
function content(n, title, bullets, opts = {}) {
  const s = p.addSlide();
  s.background = { color: WHITE };
  s.addShape(p.ShapeType.rect, { x: 0.5, y: 0.55, w: 0.16, h: 0.5, fill: { color: NAVY } });
  s.addText(title, { x: 0.8, y: 0.45, w: 12, h: 0.7, fontFace: TITLE_FONT, fontSize: 30, bold: true, color: NAVY, align: "left" });
  const items = bullets.map((b) => ({
    text: b.t !== undefined ? b.t : b,
    options: { bullet: { code: "2022", indent: 18 }, indentLevel: b.lvl || 0,
      fontSize: b.size || 17, color: DARK, paraSpaceAfter: 10 },
  }));
  s.addText(items, { x: 0.85, y: 1.45, w: 11.7, h: 5.2, fontFace: BODY_FONT, valign: "top", align: "left" });
  if (opts.note) s.addText(opts.note, { x: 0.85, y: 6.45, w: 11.7, h: 0.5, fontFace: BODY_FONT, italic: true, fontSize: 12.5, color: ACCENT });
  footer(s, n);
  return s;
}

// 1 — Title (dark)
let s = p.addSlide(); s.background = { color: NAVY };
s.addText("Real-Time Korean Sign Language\nTranslation on a Raspberry Pi", { x: 0.9, y: 2.1, w: 11.5, h: 1.8, fontFace: TITLE_FONT, fontSize: 40, bold: true, color: WHITE, align: "left", lineSpacingMultiple: 1.05 });
s.addText("with LLM-based Natural-Language Sentence Generation", { x: 0.95, y: 3.95, w: 11.5, h: 0.6, fontFace: BODY_FONT, fontSize: 20, italic: true, color: ICE });
s.addText([{ text: "Internet of Things (IoT) Systems  ·  Spring 2026  ·  HUFS", options: { breakLine: true } },
           { text: "Lee Sungjoon (202102467)  ·  Bae Jingyu (202001647)", options: {} }],
  { x: 0.95, y: 5.4, w: 11.5, h: 1, fontFace: BODY_FONT, fontSize: 15, color: ICE, lineSpacingMultiple: 1.3 });

// 2 — Problem
content(2, "The Problem", [
  "~400,000 deaf people in Korea; KSL is their primary language",
  "Very few hearing people understand KSL → daily communication barrier",
  "Prior work mostly stops at word-level recognition",
  "Goal: end-to-end words → natural sentence → speech, on a low-cost edge device",
]);

// 3 — Concept
content(3, "Concept — One-Line Pipeline", [
  "Camera → MediaPipe (2 hands) → 131-dim feature → LSTM (TFLite)",
  "→ word buffer → [button | 완료 sign | 3 s silence] → Gemini 2.5 Flash",
  "→ TTS (Korean) + LCD (English)",
  { t: "Perception is local (privacy, latency); only sentence generation calls the cloud", lvl: 0 },
]);

// 4 — Architecture
content(4, "System Architecture", [
  "Presentation layer — LCD display + TTS audio",
  "Application layer — sentence buffer + LLM controller (persona)",
  "AI / ML layer — MediaPipe → LSTM (TFLite) inference",
  "Hardware layer — Raspberry Pi 4B, camera, GPIO buttons, buzzer",
]);

// 5 — Feature
content(5, "Two-Hand 131-Dim Feature", [
  "KSL is two-handed → single-hand (63-d) input loses meaning",
  "131 = [ L 21×3 | R 21×3 | wrist-to-wrist 3 | presence 2 ]",
  "Per-hand intra-hand scale normalization",
  { t: "→ invariant to camera distance, hand size, and coordinate units", lvl: 1 },
]);

// 6 — Train/serve
content(6, "Train / Serve Consistency by Construction", [
  "Inference and dataset conversion share ONE module (feature_format.py)",
  "The same code assembles the vector in both paths",
  "→ preprocessing-level train/serve skew is structurally impossible",
]);

// 7 — Axis alignment
content(7, "Empirical Axis Alignment  (Key Contribution)", [
  "AI-Hub keypoints (meters, non-mirrored) vs MediaPipe ([0,1], mirrored)",
  "Compared the SAME clip frame-by-frame (725 pairs)",
  "Fix = x-axis flip;  correlation  x = 0.954,  y = 0.982,  z = 0.577",
  "Without it: x = −0.954 → a left-right-flipped model",
  "Plus a (w, h, w) isotropy correction for per-axis normalization",
]);

// 8 — Temporal
content(8, "Temporal Resampling  (Low-FPS Robustness)", [
  "On-device MediaPipe ≈ 8–9 FPS → a 30-frame buffer stretches 1 s into ~4 s",
  "Keep a (timestamp, vector) buffer",
  "Resample the last 1.0 s into 30 points → matches the 30-FPS training window",
]);

// 9 — Eval
content(9, "Leakage-Corrected Evaluation", [
  "Random frame split gave a misleading 1.0000 (windows / augments leaked)",
  "Switched to a held-out-signer split",
  "Held-out signers' augmented variants excluded from train AND test",
  "Only unseen-signer accuracy is used for judgment",
]);

// 10 — LLM
content(10, "LLM Sentence Generation + Persona", [
  "Word buffer → Gemini 2.5 Flash with a persona system prompt",
  "Persona: polite / friendly / brief",
  "Async worker — the camera loop never blocks",
  "Offline fallback = plain word concatenation",
]);

// 11 — Hardware
content(11, "Hardware & Accessibility", [
  "USB webcam, I2C LCD 20×4, active buzzer, 4 push buttons",
  "Buttons = complete + 3 personas; internal pull-ups to GND (no resistors)",
  "Beep 1 / 2 / 3 times = non-visual persona confirmation",
  { t: "Designed for deaf / hard-of-hearing users who cannot rely on audio cues", lvl: 0 },
]);

// 12 — Software
content(12, "Software Stack", [
  "CV: MediaPipe Hands (<0.10.30), OpenCV",
  "Edge: tflite-runtime, Python 3.11 (uv); camera via rpicam-vid on RPi",
  "Train: TensorFlow 2.15.1 on a GPU server",
  "LLM: google-genai (Gemini 2.5 Flash)",
]);

// 13 — Pitfalls
content(13, "Engineering Pitfalls Solved", [
  "LabelEncoder Unicode sort broke index↔label → original-order map + test guard",
  "TF ≥2.16 LSTM-TFLite fails / float16 OOM → TF 2.15.1, batch-1 convert, float32 (740 KB)",
  "Silent retrain on stale code → chain_train.sh: fetch + reset + code-marker check",
]);

// 14 — Results (stat callouts)
s = p.addSlide(); s.background = { color: WHITE };
s.addShape(p.ShapeType.rect, { x: 0.5, y: 0.55, w: 0.16, h: 0.5, fill: { color: NAVY } });
s.addText("Results", { x: 0.8, y: 0.45, w: 12, h: 0.7, fontFace: TITLE_FONT, fontSize: 30, bold: true, color: NAVY });
const cards = [["0.94", "held-out accuracy\n(deployed TFLite, 2 unseen signers)"], ["27 / 30", "words reliable\non the physical device"], ["740 KB", "TFLite model size\n(float32)"], ["6.4 FPS", "end-to-end on RPi 4B\n(target ≥20 — Future Work)"]];
cards.forEach((c, i) => {
  const x = 0.85 + (i % 2) * 6.1, y = 1.7 + Math.floor(i / 2) * 2.5;
  s.addShape(p.ShapeType.roundRect, { x, y, w: 5.7, h: 2.2, fill: { color: "F2F5FB" }, line: { color: ICE, width: 1 }, rectRadius: 0.1 });
  s.addText(c[0], { x: x + 0.2, y: y + 0.25, w: 5.3, h: 0.9, fontFace: TITLE_FONT, fontSize: 44, bold: true, color: NAVY, align: "left" });
  s.addText(c[1], { x: x + 0.22, y: y + 1.25, w: 5.3, h: 0.8, fontFace: BODY_FONT, fontSize: 14, color: MUTED, align: "left" });
});
footer(s, 14);

// 15 — Integrity
content(15, "Evaluation Integrity — Keras vs TFLite", [
  "Same 2,595 inputs: Keras 0.71 (CPU = GPU) vs deployed TFLite 0.94",
  "Models disagree on 30% of samples",
  "On those disagreements: TFLite correct 650 vs Keras 59",
  { t: "RPi runs the TFLite file → 0.94 is the faithful deployed number (reported with caveats)", lvl: 0 },
]);

// 16 — Limitations
content(16, "Limitations (Honest)", [
  "Held-out = 2 signers → wide confidence interval",
  "밥 / 배고프다 / 주다 still confusable on-device",
  "≥20 FPS target not met (MediaPipe bottleneck) → Future Work",
]);

// 17 — Contributions
content(17, "Contributions", [
  "Reusable dataset↔runtime alignment verification tool",
  "Single-source preprocessing (no train/serve skew)",
  "Leakage-corrected signer-level evaluation; honest accuracy trajectory",
  "Accessibility-first control; automated cloud-train → edge-deploy",
]);

// 18 — Future / Demo (dark closing)
s = p.addSlide(); s.background = { color: NAVY };
s.addText("Future Work & Demo", { x: 0.9, y: 0.7, w: 11.5, h: 0.9, fontFace: TITLE_FONT, fontSize: 34, bold: true, color: WHITE });
s.addText([
  { text: "Continuous signing (SEN dataset)", options: { bullet: { code: "2022" }, breakLine: true } },
  { text: "Vocabulary scaling (30 → hundreds)", options: { bullet: { code: "2022" }, breakLine: true } },
  { text: "Upper-body pose keypoints for position-dependent signs", options: { bullet: { code: "2022" }, breakLine: true } },
  { text: "On-device LLM (offline); higher-FPS path", options: { bullet: { code: "2022" }, breakLine: true } },
], { x: 1.0, y: 2.0, w: 11, h: 2.6, fontFace: BODY_FONT, fontSize: 18, color: ICE, lineSpacingMultiple: 1.35 });
s.addText("Live / recorded demo  →  Q&A", { x: 1.0, y: 5.2, w: 11, h: 0.8, fontFace: TITLE_FONT, fontSize: 24, bold: true, color: WHITE });
s.addText("Thank you", { x: 1.0, y: 6.1, w: 11, h: 0.6, fontFace: BODY_FONT, fontSize: 16, italic: true, color: ICE });

p.writeFile({ fileName: "/tmp/ksl-review/docs/KSL-LLM-IoT_Slides.pptx" }).then((f) => console.log("WROTE", f));
