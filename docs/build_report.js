const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, Header, Footer, PageBreak, TabStopType, TabStopPosition,
} = require("docx");

const CONTENT_W = 9360; // US Letter, 1" margins
const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };
const HEAD_FILL = "D5E8F0";

function P(text, opts = {}) {
  return new Paragraph({ spacing: { after: 120 }, ...opts,
    children: [new TextRun({ text, ...(opts.run || {}) })] });
}
function runs(children, opts = {}) {
  return new Paragraph({ spacing: { after: 120 }, ...opts, children });
}
function H1(text) { return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] }); }
function H2(text) { return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] }); }
function bullet(text) { return new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 60 }, children: [new TextRun(text)] }); }
function num(text) { return new Paragraph({ numbering: { reference: "n", level: 0 }, spacing: { after: 60 }, children: [new TextRun(text)] }); }

function cell(text, w, { headerCell = false, bold = false } = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: headerCell ? { fill: HEAD_FILL, type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({ children: [new TextRun({ text, bold: bold || headerCell, size: 20 })] })],
  });
}
function table(colW, rows) {
  return new Table({
    width: { size: colW.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: colW,
    rows: rows.map((r, i) => new TableRow({
      children: r.map((c) => cell(c, colW[r.indexOf(c)] || colW[0], { headerCell: i === 0 })),
    })),
  });
}
// simpler table builder mapping each cell by index
function tbl(colW, rows) {
  return new Table({
    width: { size: colW.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: colW,
    rows: rows.map((r, ri) => new TableRow({
      children: r.map((txt, ci) => cell(txt, colW[ci], { headerCell: ri === 0 })),
    })),
  });
}

const children = [];

// ---- Title block ----
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1200, after: 80 },
  children: [new TextRun({ text: "Real-Time Korean Sign Language Translation on a Raspberry Pi", bold: true, size: 40 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
  children: [new TextRun({ text: "with LLM-based Natural-Language Sentence Generation", italics: true, size: 26 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: "Internet of Things (IoT) Systems — Spring 2026, HUFS", size: 24 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [new TextRun({ text: "Lee Sungjoon (202102467) · Bae Jingyu (202001647)", size: 24 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "Submission: 2026-06-22", size: 24 })] }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ---- Abstract ----
children.push(H1("Abstract"));
children.push(P("KSL-LLM-IoT is a real-time Korean Sign Language (KSL) translator that runs on a Raspberry Pi 4B edge device. A camera captures two-hand signing; MediaPipe extracts 3D hand landmarks; a TensorFlow-Lite LSTM classifies 30 KSL words; and a large language model (Gemini 2.5 Flash) composes the recognized word sequence into a natural Korean sentence, delivered through a speaker (TTS) and an I2C LCD. Physical buttons provide an accessible, latency-free control interface. On a held-out set of two unseen signers the deployed TFLite model reaches 0.94 word-classification accuracy; on the physical device 27 of 30 words are recognized reliably. The main engineering contributions are (i) an empirically verified coordinate-system alignment between a public keypoint dataset and the runtime camera, (ii) a single-source feature representation that structurally eliminates train/serve skew, and (iii) a leakage-corrected, signer-level evaluation protocol."));

// ---- 1 Project Idea ----
children.push(H1("1. Project Idea"));
children.push(runs([new TextRun({ text: "Problem. ", bold: true }), new TextRun("Roughly 400,000 deaf people in Korea use KSL as their primary language, yet very few hearing people understand it, creating a daily communication barrier that today depends on scarce human interpreters. Most prior KSL work stops at word-level classification; an end-to-end pipeline that produces natural sentences and speech on an edge device is rare.")]));
children.push(runs([new TextRun({ text: "Concept. ", bold: true }), new TextRun("A low-cost Raspberry Pi performs all perception locally (privacy, low latency) and calls a cloud LLM only for the final sentence-generation step:")]));
children.push(P("Camera → MediaPipe (2 hands) → 131-dim feature → LSTM (TFLite) → word buffer → [complete button | 완료 sign | 3 s silence] → Gemini 2.5 Flash → TTS (Korean) + LCD (English).", { run: { italics: true } }));
children.push(runs([new TextRun({ text: "IoT characteristics. ", bold: true }), new TextRun("Sensors/actuators (camera, GPIO buttons, buzzer) → on-device inference → outputs (LCD, speaker); plus a cloud-train → edge-deploy lifecycle.")]));
children.push(P("Differentiators:", { run: { bold: true } }));
children.push(bullet("Natural LLM sentences rather than a word list."));
children.push(bullet("User-selectable sentence persona (polite / friendly / brief)."));
children.push(bullet("Accessibility-first control: physical buttons + non-visual beep feedback for deaf / hard-of-hearing users."));
children.push(bullet("A measured train/serve coordinate alignment between the AI-Hub dataset and the runtime camera."));

// ---- 2 System Design ----
children.push(H1("2. System Design & Methodology"));
children.push(H2("2.1 Pipeline (Fig.1)"));
children.push(P("Camera → MediaPipe (2 hands) → 131-dim feature → LSTM (TFLite) → word buffer → trigger → Gemini → TTS + LCD."));
children.push(H2("2.2 Two-Hand 131-Dim Representation (Fig.2)"));
children.push(P("KSL is a two-handed language, so a single-hand (63-dim) input cannot preserve meaning. The feature is 131 = [ LEFT 21×3 | RIGHT 21×3 | wrist-to-wrist vector 3 | presence flags 2 ]. Each hand is normalized by its wrist-relative coordinates divided by an intra-hand scale (‖landmark9 − landmark0‖), making it invariant to camera distance, hand size, and coordinate units. A missing hand is zero-filled with its presence flag cleared."));
children.push(H2("2.3 Train/Serve Consistency by Construction"));
children.push(P("Inference (hand_tracker.py) and dataset conversion (convert_aihub.py) share a single module, feature_format.py. Because the exact same code assembles the 131-dim vector in both paths, preprocessing-level train/serve skew is structurally impossible."));
children.push(H2("2.4 Empirical Axis Alignment (Fig.3)"));
children.push(P("AI-Hub provides multi-view 3D-reconstructed keypoints (meters, non-mirrored); MediaPipe provides normalized [0,1] coordinates from a mirrored selfie image. We compared the MP4 (runtime path) and the keypoints of the same clip frame-by-frame (725 pairs):"));
children.push(bullet("Required transform = x-axis sign flip (AIHUB_AXIS_SIGNS = (-1, 1, 1))."));
children.push(bullet("Correlation after transform: x = 0.954, y = 0.982, z = 0.577 (without it, x = −0.954 → a left-right-flipped model)."));
children.push(bullet("z = 0.577 reflects the limits of monocular depth but is a directionally consistent auxiliary signal (no ToF camera needed)."));
children.push(bullet("An additional (w, h, w) isotropy correction aligns MediaPipe's per-axis normalization to the dataset's isotropic metric coordinates."));
children.push(H2("2.5 Classifier and Temporal Normalization"));
children.push(P("2×LSTM (128, 64) + Dense, input (30, 131), 30 classes, exported to a 740 KB float32 TFLite model. Because on-device MediaPipe runs at only ~8–9 FPS, a fixed 30-frame buffer would stretch a 1-second sign into a ~4-second window. The classifier instead keeps a (timestamp, vector) buffer and linearly resamples the most recent 1.0 s into 30 points, restoring the training-time temporal window."));
children.push(H2("2.6 Leakage-Corrected, Signer-Level Evaluation"));
children.push(P("An early model evaluated with a random frame-level split reported a misleading 1.0000 test accuracy because sliding windows and augmentations of the same clip leaked across train/test. We switched to a held-out-signer split (model/data_split.py, holdout.json): two signers are held out entirely, and their augmented variants are excluded from both train and test. The held-out accuracy of unseen signers is the only number used for judgment."));
children.push(H2("2.7 LLM Sentence Generation"));
children.push(P("The word buffer is sent to Gemini 2.5 Flash with a persona-specific system prompt; the call runs in an async worker so the camera loop never blocks, and falls back to a plain word concatenation when offline."));

// ---- 3 Hardware & Software ----
children.push(H1("3. Hardware & Software Details"));
children.push(H2("3.1 Hardware (Fig.4)"));
children.push(tbl([5000, 2360, 2000], [
  ["Component", "Interface", "Pin"],
  ["USB webcam (/dev/video0, demo) — Pi Camera v1/CSI also supported", "USB / CSI", "—"],
  ["I2C LCD 20×4 (0x27)", "I2C", "GPIO2/3"],
  ["Active buzzer", "GPIO out", "GPIO17"],
  ["Push buttons ×4 (complete + persona×3)", "GPIO in (pull-up)", "GPIO5/6/13/19 ↔ GND"],
  ["Speaker", "3.5mm / USB", "separate power"],
]));
children.push(P("Buttons use internal pull-ups switching to GND (~66 µA, no external resistor). The buzzer beeps 1/2/3 times to confirm the selected persona without looking at a screen.", { spacing: { before: 120, after: 120 } }));
children.push(H2("3.2 Software Stack"));
children.push(tbl([2400, 4000, 2960], [
  ["Layer", "Technology", "Notes"],
  ["Computer vision", "MediaPipe Hands (<0.10.30), OpenCV", "legacy solutions API"],
  ["Edge inference", "tflite-runtime", "RPi, Python 3.11 via uv"],
  ["Training", "TensorFlow 2.15.1 (GPU)", "cloud server (RTX 4000 Ada)"],
  ["LLM", "google-genai (Gemini 2.5 Flash)", "persona system prompt, async"],
  ["OS / IoT", "RPi OS Trixie, RPi.GPIO (polling), smbus2", "rpicam-vid camera backend"],
]));
children.push(H2("3.3 Engineering Pitfalls Solved"));
children.push(bullet("LabelEncoder bug: scikit-learn sorted labels by Unicode, breaking the index↔label mapping → fixed with an original-order LABEL_TO_IDX plus a smoke-test guard."));
children.push(bullet("TFLite conversion: TF ≥2.16 cannot convert the LSTM (MLIR bug) and float16 quantization OOMs → pinned TF 2.15.1, batch-size-1 conversion, float32 (model only 740 KB)."));
children.push(bullet("Silent training failure: an exit-0 retrain had silently trained on stale code (a dirty server tree made git pull fail without error) → chain_train.sh uses fetch + reset --hard + a code-marker grep before training."));

// ---- 4 Results ----
children.push(H1("4. Results & Contributions"));
children.push(H2("4.1 Quantitative Results"));
children.push(tbl([6360, 3000], [
  ["Item", "Value"],
  ["Pipeline validation (example 18 classes, 3,540 samples)", "TFLite accuracy 0.98"],
  ["Deployed model (30 classes, 16 signers) — held-out (2 unseen signers, 2,595 seq)", "TFLite accuracy 0.94"],
  ["Physical device (USB webcam, diagnose_live)", "27/30 words reliable"],
  ["TFLite inference latency", "0.41 ms (server CPU) / not yet measured (RPi)"],
  ["End-to-end FPS on RPi 4B", "6.4 — target ≥20 not met (Future Work)"],
  ["Model size", "740 KB"],
  ["Sentence latency (Gemini)", "not yet measured (target ≤ 4 s)"],
]));
children.push(H2("4.2 On the Keras-vs-TFLite Gap (Evaluation Integrity)"));
children.push(P("The Keras checkpoint scores 0.71 when run as a Keras model, while the deployed TFLite scores 0.94 on the same 2,595 held-out inputs. We measured this directly: Keras = 0.7129 on both CPU and GPU (so it is not a device/cuDNN artifact); the two models disagree on 775/2,595 (29.9%) of samples, and on those disagreements TFLite is correct 650 times vs Keras 59. The Raspberry Pi runs exactly the TFLite file, so we report 0.94 as the deployed model's held-out accuracy, while disclosing that (a) it is measured on two held-out signers, and (b) live performance is 27/30 due to the runtime domain gap."));
children.push(H2("4.3 Honest Limitations"));
children.push(P("Accuracy is reported on two held-out signers (wide confidence interval); recognition of 밥/배고프다/주다 remains confusable on-device; and the FPS target is not met. These are discussed rather than hidden."));
children.push(H2("4.4 Contributions"));
children.push(num("A reusable methodology and tool (verify_aihub_alignment.py) for empirically verifying coordinate alignment between a public keypoint dataset and a runtime extractor."));
children.push(num("Single-source feature preprocessing that structurally prevents train/serve skew."));
children.push(num("A leakage-corrected, signer-level evaluation that documents the honest accuracy trajectory (1.0000 leaked → 0.36 single-signer → 0.71/0.94 with 16 signers)."));
children.push(num("Accessibility-first interface: deterministic physical buttons with non-visual beep feedback."));
children.push(num("A fully automated cloud-train → edge-deploy pipeline."));

// ---- 5 Future Work ----
children.push(H1("5. Future Work"));
children.push(bullet("Continuous-signing recognition using the AI-Hub sentence (SEN) dataset."));
children.push(bullet("Vocabulary scaling (30 → hundreds); the label↔headword pipeline already generalizes."));
children.push(bullet("Add upper-body pose keypoints to separate position-dependent signs (나 vs 당신)."));
children.push(bullet("On-device lightweight LLM (e.g., Gemma) for fully offline operation."));
children.push(bullet("Higher-FPS path (MediaPipe Tasks API) to close the ≥20 FPS gap."));
children.push(bullet("Korean-capable graphic LCD or a mobile companion app."));

// ---- References ----
children.push(H1("References"));
children.push(num("Shin, J. et al. (2023). Dynamic KSL Recognition. IEEE Access 11."));
children.push(num("Miah, A. S. M. et al. (2023). KSL Recognition Using a Transformer DNN. Applied Sciences 13(5)."));
children.push(num("Sánchez-Vicinaiz, T. J. et al. (2024). MediaPipe + CNN on Raspberry Pi. Technologies 12(8)."));
children.push(num("AI-Hub Korean Sign Language video dataset (keypoints)."));

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: "1F3864" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Arial", color: "2E5496" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 280 } } } }] },
      { reference: "n", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 280 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "KSL-LLM-IoT — IoT Systems Term Project    ", size: 18, color: "888888" }), new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 18, color: "888888" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("/tmp/ksl-review/docs/KSL-LLM-IoT_Report.docx", buf);
  console.log("WROTE /tmp/ksl-review/docs/KSL-LLM-IoT_Report.docx", buf.length, "bytes");
});
