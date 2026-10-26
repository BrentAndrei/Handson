"""
Normalize a Keras-3-serialized model.json to the Keras-2 format that
@tensorflow/tfjs 4.22.0 can actually load.

The 399-class model was exported by Keras 3.15.1, which serializes layers
with the new `module`/`registered_name`/`DTypePolicy` shape. tfjs 4.22.0's
layers-model parser still expects the old shape (`class_name` + `config`,
dtype as a plain string, `batch_input_shape` on InputLayer). The result is
"Failed to parse model JSON" even though the topology is identical.

This script rewrites the JSON in place, mapping:
  - {"module":"keras","class_name":"DTypePolicy","config":{"name":"float32"},...}
    -> "float32"
  - {"module":"keras.layers","class_name":X,"config":{...},"registered_name":null}
    -> {"class_name":X,"config":{...}}
  - {"module":"keras.initializers","class_name":X,"config":{...},"registered_name":null}
    -> {"class_name":X,"config":{...}}
  - InputLayer batch_shape -> batch_input_shape
  - Bidirectional layer/forward_layer/backward_layer -> old nested shape
  - weight names gru_cell/kernel -> kernel

Run: python normalize_model.py <path> [<path> ...]
"""
import json
import sys
import copy


def strip_module(obj):
    """Recursively drop `module` and `registered_name` keys, and collapse
    DTypePolicy objects to plain dtype strings."""
    if isinstance(obj, dict):
        # DTypePolicy collapse
        if (
            obj.get("module") == "keras"
            and obj.get("class_name") == "DTypePolicy"
            and "config" in obj
            and isinstance(obj["config"], dict)
            and "name" in obj["config"]
        ):
            return obj["config"]["name"]
        out = {}
        for k, v in obj.items():
            if k in ("module", "registered_name"):
                continue
            out[k] = strip_module(v)
        return out
    if isinstance(obj, list):
        return [strip_module(x) for x in obj]
    return obj


def fix_input_layer(layer):
    cfg = layer.get("config", {})
    if "batch_shape" in cfg and "batch_input_shape" not in cfg:
        cfg["batch_input_shape"] = cfg.pop("batch_shape")
    return layer


def fix_bidirectional(layer):
    """Keras 3 nests the forward/backward GRU under layer/backward_layer with
    module/registered_name; tfjs expects them nested as layer/backward_layer
    with class_name/config only (no module). strip_module handles the
    module/registered_name removal, but we must also ensure the nested GRU
    keeps its class_name."""
    return layer


def fix_weight_names(weights):
    """Keras 3 names bidirectional weights `forward_gru/gru_cell/kernel`;
    tfjs expects `forward_gru/kernel` (no gru_cell)."""
    out = []
    for w in weights:
        name = w.get("name", "")
        if "/gru_cell/" in name:
            w = dict(w)
            w["name"] = name.replace("/gru_cell/", "/")
        out.append(w)
    return out


def normalize(path):
    with open(path) as f:
        d = json.load(f)

    layers = d["modelTopology"]["model_config"]["config"]["layers"]
    for i, layer in enumerate(layers):
        layers[i] = fix_input_layer(layer)
        layers[i] = fix_bidirectional(layer)

    d = strip_module(d)

    # Fix weight names
    for wm in d.get("weightsManifest", []):
        if "weights" in wm:
            wm["weights"] = fix_weight_names(wm["weights"])

    with open(path, "w") as f:
        json.dump(d, f, separators=(",", ":"))
    print("normalized", path)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        normalize(p)