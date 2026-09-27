const fs = require('fs');
const tf = require('@tensorflow/tfjs');

async function testLoad() {
  try {
    const modelJson = JSON.parse(fs.readFileSync('public/models/fsl_model/model.json', 'utf8'));
    const binBuf = fs.readFileSync('public/models/fsl_model/group1-shard1of1.bin');

    const handler = {
      load() {
        return Promise.resolve({
          modelTopology: modelJson.modelTopology,
          weightSpecs: modelJson.weightsManifest[0].weights,
          weightData: binBuf.buffer.slice(0),
        });
      },
    };

    const model = await tf.loadLayersModel(handler);
    console.log('SUCCESS!');
    model.dispose();
  } catch (err) {
    console.error('ERROR:', err.constructor.name, '-', err.message);
    // Get more detail
    const errStr = err.toString();
    console.error('Full error:', errStr.substring(0, 500));
  }
}

testLoad();
