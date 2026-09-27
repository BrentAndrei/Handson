const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs');

async function testLoad() {
  try {
    console.log('tfjs version:', tf.version.tfjs);

    const modelPath = path.resolve('public/models/fsl_model/model.json');
    const binPath = path.resolve('public/models/fsl_model/group1-shard1of1.bin');

    const modelJson = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
    const binBuffer = fs.readFileSync(binPath);

    const handler = tf.io.fromMemory(
      modelJson.model_config.config.name || 'model',
      modelJson,
      [binBuffer]
    );

    console.log('Loading model from memory...');
    const model = await tf.loadLayersModel(handler);
    console.log('Model loaded successfully!');
    console.log('Model inputs:', model.inputs.map(i => i.name));
    console.log('Model outputs:', model.outputs.map(o => o.name));
    model.dispose();
  } catch (err) {
    console.error('Error loading model:', err.message);
    console.error('Stack:', err.stack?.substring(0, 1000));
  }
}

testLoad();
