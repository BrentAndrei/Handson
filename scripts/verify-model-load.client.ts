import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

(async () => {
  try {
    console.log("backend:", tf.getBackend());
    await tf.setBackend("webgl");
    await tf.ready();
    console.log("backend ready:", tf.getBackend());

    const m = await tf.loadLayersModel("/models/fsl_model/model.json");
    console.log("LOADED OK inputs:", m.inputs.map(x => x.shape), "outputs:", m.outputs.map(x => x.shape));

    const t = tf.zeros([1, 16, 345]);
    const p = m.predict(t) as tf.Tensor;
    console.log("PREDICT OK shape:", p.shape);
    p.dispose(); t.dispose(); m.dispose();
  } catch (e) {
    console.error("FAILED:", e instanceof Error ? e.message : String(e));
    if (e instanceof Error && e.stack) console.error(e.stack);
  }
})();