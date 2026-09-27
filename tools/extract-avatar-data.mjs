import { createReadStream } from 'fs';
import { writeFileSync, existsSync, mkdirSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import JSONStream from 'JSONStream';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const INPUT = join(ROOT, 'training', 'combined_dataset_filtered.json');
const OUTPUT_DIR = join(ROOT, 'public', 'avatar-data');

if (!existsSync(OUTPUT_DIR)) mkdirSync(OUTPUT_DIR, { recursive: true });

function sanitizeFilename(name) {
  return name.replace(/[^a-zA-Z0-9]/g, '_');
}

async function extract() {
  console.log('Streaming', INPUT, '...');

  const labelFrames = new Map();
  let totalSamples = 0;
  const labelCounts = {};

  const stream = createReadStream(INPUT, { encoding: 'utf8' });
  const parser = JSONStream.parse('samples.*');

  parser.on('data', (sample) => {
    const label = sample.label;
    const frames = sample.frames;
    if (!label || !Array.isArray(frames) || frames.length === 0) return;

    if (!labelFrames.has(label)) {
      labelFrames.set(label, []);
      labelCounts[label] = 0;
    }
    labelFrames.get(label).push(frames);
    labelCounts[label] += frames.length / 345;
    totalSamples++;

    if (totalSamples % 5000 === 0) {
      console.log(`  ${totalSamples} samples, ${labelFrames.size} labels`);
    }
  });

  parser.on('end', () => {
    console.log(`Extracted ${totalSamples} samples, ${labelFrames.size} labels`);

    let written = 0;
    for (const [label, framesList] of labelFrames) {
      const fname = sanitizeFilename(label) + '.json';
      const outPath = join(OUTPUT_DIR, fname);
      const allFrames = framesList.flat();
      const data = {
        label,
        frameCount: Math.round(allFrames.length / 345),
        frames: allFrames,
      };
      try {
        writeFileSync(outPath, JSON.stringify(data));
        written++;
      } catch (e) {
        console.error('Write error for', label, e.message);
      }
    }

    const manifest = {
      generatedAt: new Date().toISOString(),
      source: INPUT,
      totalLabels: written,
      totalFrames: Math.round(
        Array.from(labelFrames.values()).flat().length / 345
      ),
      labels: Object.keys(labelCounts).sort(),
      labelFrameCounts: labelCounts,
    };
    writeFileSync(join(OUTPUT_DIR, 'manifest.json'), JSON.stringify(manifest, null, 2));

    console.log(`Wrote ${written} label files`);
  });

  stream.pipe(parser);
}

extract().catch((e) => {
  console.error('Extraction failed:', e);
  process.exit(1);
});
