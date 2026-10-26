const fs = require('fs');
const { sliceIntoWindows } = require('./src/features/windowBuffer');
const { FEATURE_DIM } = require('./src/features/featureSelect');
const { DEFAULT_STRIDE } = require('./src/features/windowBuffer');
const INFERENCE_LABELS = require('./src/inference/labels').INFERENCE_LABELS;
const ROOT = 'public/avatar-data';
const TOTAL_FLOATS = 345;
function readFrames(label) {
  const cands = [encodeURIComponent(label) + '.json', label.replace(/[^a-zA-Z0-9]/g, '_') + '.json'];
  for (const c of cands) {
    const p = path.join(ROOT, c);
    if (!fs.existsSync(p)) continue;
    const d = JSON.parse(fs.readFileSync(p, 'utf-8'));
    if (!Array.isArray(d.frames)) continue;
    const flat = new Float32Array(d.frames);
    const n = Math.floor(flat.length / TOTAL_FLOATS);
    const frames = [];
    for (let i=0;i<n;i++) frames.push(flat.slice(i*TOTAL_FLOATS, (i+1)*TOTAL_FLOATS));
    return frames;
  }
  return null;
}
async function test(label) {
  const raw = readFrames(label);
  console.log(label, 'raw frames', raw?.length);
  if (!raw || raw.length === 0) return;
  const feats = raw;
  const windows = sliceIntoWindows(feats, FEATURE_DIM, 16, DEFAULT_STRIDE);
  console.log('  windows', windows.length);
  for (let i=0;i<Math.min(10,windows.length);i++) {
    const w = windows[i];
    const top = w.indexOf(Math.max(...w));
    console.log('    w', i, 'top label:', INFERENCE_LABELS[top]);
  }
  // Now test with temporal ensemble
  const ensemble = new (require('./src/inference/temporalEnsemble').TemporalEnsemble)(INFERENCE_LABELS.length, INFERENCE_LABELS);
  for (const w of windows) {
    const probs = new Float32Array(w);
    const result = ensemble.push(probs);
    if (result) {
      console.log('      ENS Emit:', result.label, 'confidence:', result.confidence.toFixed(3), 'margin:', (result.margin||0).toFixed(3));
    }
  }
  console.log('  final ring size:', ensemble.ring.length);
  console.log('  emissions count:', ensemble.ring.length > 0 ? 'some' : 'none');
}
test('THANK_YOU');
test('ARE_YOU_DEAF');