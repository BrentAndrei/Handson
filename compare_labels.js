const fs = require('fs');
const appContent = fs.readFileSync('src/App.tsx', 'utf8');
const match = appContent.match(/const INFERENCE_LABELS: string\[\] = \[([\s\S]*?)\];/);
const labels = match[1].match(/"([^"]+)"/g).map(function(s) { return s.slice(1, -1); });
console.log('App count:', labels.length);

const lm = JSON.parse(fs.readFileSync('training/model_compiled_consolidated/label_map.json', 'utf8'));
console.log('Model count:', lm.labels.length);

const a = new Set(labels);
const mo = new Set(lm.labels);
const onlyApp = [...a].filter(function(x) { return !mo.has(x); });
const onlyModel = [...mo].filter(function(x) { return !a.has(x); });
console.log('Only in app:', onlyApp);
console.log('Only in model:', onlyModel);
console.log('Same order:', labels.every(function(l, i) { return l === lm.labels[i]; }));

// Find duplicates in app
const seen = {};
const dupes = [];
labels.forEach(function(l) { if (seen[l]) dupes.push(l); seen[l] = 1; });
console.log('App duplicates:', dupes);
