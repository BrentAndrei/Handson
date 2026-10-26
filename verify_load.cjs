const tf = require('@tensorflow/tfjs');
(async () => {
  try {
    const m = await tf.loadLayersModel('http://127.0.0.1:5173/models/fsl_model/model.json');
    console.log('LOADED OK  inputs:', m.inputs.map(x => x.shape), ' outputs:', m.outputs.map(x => x.shape));
    const t = tf.zeros([1, 16, 345]);
    const p = m.predict(t);
    console.log('PREDICT OK  output shape:', p.shape);
    p.dispose(); t.dispose(); m.dispose();
  } catch (e) { console.error('FAILED:', e.message); }
})();