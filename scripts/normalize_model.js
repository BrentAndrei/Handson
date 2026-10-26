/**
 * normalize_model.js — rewrite a TF.js layers-model (Keras 3 / tfjs-converter
 * v3.18.0) so it loads with installed @tensorflow/tfjs@4.22.0.
 *
 * Keras-3 serializes every layer as {module, class_name, config, registered_name}.
 * tfjs@4 (Keras-2 format) expects {class_name, config}. This script:
 *   1. Collapses DTypePolicy objects {module, class_name, config:{name}} -> "name"
 *   2. Rewraps Keras-3 {module, class_name, config} -> {class_name, config}
 *   3. Fixes InputLayer batch_shape -> batch_input_shape
 *   4. Converts inbound_nodes from Keras-3 {args,kwargs} -> Keras-2 [[name,i,j,{}]]
 *   5. Rewrites GRU weight names forward_gru/gru_cell/* -> forward_gru/*
 *
 * Usage:  node scripts/normalize_model.js <in.json> <out.json>
 */
import * as fs from "fs";

/** Collapse every DTypePolicy object {module, class_name, config:{name}} -> "name" */
function flattenDtypes(obj) {
  if (!obj || typeof obj !== "object") return;
  if (Array.isArray(obj)) {
    for (const v of obj) flattenDtypes(v);
    return;
  }
  if (
    obj.dtype &&
    typeof obj.dtype === "object" &&
    (obj.dtype.module === "keras" || obj.dtype.class_name === "DTypePolicy")
  ) {
    obj.dtype = obj.dtype.config?.name || obj.dtype.name || "float32";
  }
  for (const k of Object.keys(obj)) flattenDtypes(obj[k]);
}

/**
 * Rewrap Keras-3 {module, class_name, config, registered_name}
 * -> Keras-2 {class_name, config, ...otherKeys}.
 * Only triggers on objects that have BOTH module and class_name and config.
 * Preserves sibling keys such as build_config.
 */
function rewrapModules(obj) {
  if (!obj || typeof obj !== "object") return;
  if (Array.isArray(obj)) {
    for (const v of obj) rewrapModules(v);
    return;
  }
  if (obj.module && obj.class_name && obj.config) {
    const cn = obj.class_name;
    const cfg = obj.config;
    const extra = {};
    for (const k of Object.keys(obj)) {
      if (k !== "module" && k !== "registered_name" && k !== "class_name" && k !== "config") {
        extra[k] = obj[k];
      }
    }
    Object.keys(obj).forEach((k) => delete obj[k]);
    obj.class_name = cn;
    obj.config = cfg;
    Object.assign(obj, extra);
  }
  for (const k of Object.keys(obj)) rewrapModules(obj[k]);
}

/** Convert Keras-3 inbound_nodes -> Keras-2 form. */
function fixInboundNodes(layer) {
  const nodes = layer.inbound_nodes;
  if (!Array.isArray(nodes)) return;
  const out = [];
  for (const node of nodes) {
    if (Array.isArray(node)) continue; // already Keras-2 form
    const kwargs = node.kwargs || {};
    const args = Array.isArray(node.args) ? node.args : [];
    const tensors = [];
    for (const arg of args) {
      const kh = arg?.config?.keras_history;
      if (Array.isArray(kh) && kh.length >= 3) {
        tensors.push([kh[0], kh[1], kh[2], kwargs]);
      }
    }
    out.push(tensors);
  }
  layer.inbound_nodes = out;
}

/** Convert Keras-3 input_layers/output_layers -> Keras-2 form.
 * Keras-3 flattens each tensor into 3 entries: [name,i,j, name2,i2,j2, ...]
 * Keras-2 nests them: [[name,i,j,{}], [name2,i2,j2,{}], ...] */
function fixLayerList(list) {
  if (!Array.isArray(list) || list.length === 0) return;
  // Already nested form?
  if (Array.isArray(list[0])) return;
  const out = [];
  for (let i = 0; i < list.length; i += 3) {
    out.push([list[i], list[i + 1], list[i + 2], {}]);
  }
  list.length = 0;
  out.forEach((v) => list.push(v));
}

function normalize(inPath, outPath) {
  const model = JSON.parse(fs.readFileSync(inPath, "utf8"));
  const mc = model.modelTopology.model_config;

  // 1. InputLayer: batch_shape -> batch_input_shape
  for (const layer of mc.config.layers) {
    if (layer.class_name === "InputLayer" && layer.config.batch_shape) {
      layer.config.batch_input_shape = layer.config.batch_shape;
      delete layer.config.batch_shape;
    }
    fixInboundNodes(layer);
  }

  // 2. Collapse DTypePolicy objects, then rewrap module/registered_name wrappers
  flattenDtypes(model);
  rewrapModules(model);

  // 2b. Convert input_layers / output_layers from Keras-3 [name,i,j] -> Keras-2 [[name,i,j,{}]]
  fixLayerList(mc.config.input_layers);
  fixLayerList(mc.config.output_layers);

  // 2c. tfjs@4 Bidirectional reconstructs the backward layer from the forward
  // layer by flipping goBackwards — it never reads a backward_layer key.
  for (const layer of mc.config.layers) {
    if (layer.class_name === "Bidirectional" && layer.config.backward_layer) {
      delete layer.config.backward_layer;
    }
  }

  // 2d. The Bidirectional constructor prefixes the sub-layer name with
  // "forward_" (gru -> forward_gru). Keras-3 already names it forward_gru,
  // which would become forward_forward_gru and break weight lookup. Strip the
  // "forward_" prefix so the constructor produces the manifest's forward_gru.
  for (const layer of mc.config.layers) {
    if (layer.class_name === "Bidirectional" && layer.config.layer) {
      const sub = layer.config.layer;
      if (sub.config && typeof sub.config.name === "string" && sub.config.name.startsWith("forward_")) {
        sub.config.name = sub.config.name.slice("forward_".length);
      }
    }
  }

  // 3. GRU weight names: forward_gru/gru_cell/* -> forward_gru/*
  if (model.weightsManifest) {
    for (const group of model.weightsManifest) {
      for (const w of group.weights || []) {
        if (w.name.endsWith("/gru_cell/kernel")) {
          w.name = w.name.replace("/gru_cell/kernel", "/kernel");
        } else if (w.name.endsWith("/gru_cell/recurrent_kernel")) {
          w.name = w.name.replace("/gru_cell/recurrent_kernel", "/recurrent_kernel");
        } else if (w.name.endsWith("/gru_cell/bias")) {
          w.name = w.name.replace("/gru_cell/bias", "/bias");
        }
      }
    }
  }

  fs.writeFileSync(outPath, JSON.stringify(model));
  console.log(`Normalized ${inPath} -> ${outPath}`);
}

const [inPath, outPath] = process.argv.slice(2);
if (!inPath || !outPath) {
  console.error("Usage: node scripts/normalize_model.js <in.json> <out.json>");
  process.exit(2);
}
normalize(inPath, outPath);