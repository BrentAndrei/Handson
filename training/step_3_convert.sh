#!/usr/bin/env bash
# Converts the trained Keras model (training/train.py's output) into a
# TensorFlow.js Layers model with quantized weights, ready to drop into
# /public/models/fsl_model/ for the app's MODEL_URL to load.
#
# Run this in its OWN virtualenv, separate from the one used to train:
#
#   python3 -m venv .venv-convert
#   source .venv-convert/bin/activate      # .venv-convert\Scripts\activate on Windows
#   pip install tensorflowjs
#
# Why a separate env: the tensorflowjs pip package pins a specific
# tensorflow version as a hard dependency and has a history of failing
# loudly (import errors, symbol mismatches) if a different tensorflow
# version is already installed in the same environment. Installing it
# fresh lets pip resolve its own compatible tensorflow automatically
# instead of fighting the one requirements-train.txt installed.
#
# --output_format=tfjs_layers_model (not tfjs_graph_model) is the
# important flag here: it re-implements Keras layers directly in
# TensorFlow.js's JS/WebGL runtime, which is what makes the Bidirectional
# GRU layers from train.py actually work in-browser. tfjs_graph_model
# instead tries to map the SavedModel's raw ops one-for-one onto TFJS's
# op kernel set, and recurrent ops are exactly the ops that tend to be
# unsupported or behave differently there — so for this model,
# layers_model is not just an option, it's the one that reliably works.
set -euo pipefail

MODEL_IN="${1:-model_out/fsl_model.h5}"
MODEL_OUT="${2:-model_out/tfjs_model}"

# --quantize_float16 halves weight size (fp32 -> fp16) with negligible
# accuracy loss for a small GRU classifier like this — the safer default
# for a real-time app. Swap to --quantize_uint8 for a further ~4x size
# cut if bundle size matters more than the last bit of prediction
# confidence; test both against your val set before shipping uint8, since
# 8-bit quantization can measurably blur close-confusion sign pairs.
tensorflowjs_converter \
  --input_format=keras \
  --output_format=tfjs_layers_model \
  --quantize_float16=* \
  "$MODEL_IN" \
  "$MODEL_OUT"

echo "Converted model written to $MODEL_OUT"
echo "Copy it to handson/public/models/fsl_model/ (or update MODEL_URL in App.tsx to point elsewhere)."
echo "Also copy model_out/label_map.json's \"labels\" array into INFERENCE_LABELS in App.tsx."

# --- INT8 alternative (uncomment to use instead of float16) ---
# tensorflowjs_converter \
#   --input_format=keras \
#   --output_format=tfjs_layers_model \
#   --quantize_uint8=* \
#   "$MODEL_IN" \
#   "$MODEL_OUT"
