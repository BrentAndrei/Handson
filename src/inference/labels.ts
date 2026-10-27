/**
 * The classifier's label vocabulary, in the EXACT order the model's
 * output layer was trained with.
 *
 * This list lived inline in App.tsx until the capture studio needed
 * it too (to test a captured clip against the classifier before
 * installing it). The order is a contract: index i of a prediction
 * vector is this list's i-th label, so it must never be sorted,
 * filtered, or deduplicated. `label_map.json` does not ship with
 * the exported model, so this array is the source of truth.
 *
 * 54-label vocabulary of the retrained BiGRU (fsl_bigru_v2_all54).
 * Trained on Kaggle FSL (28) + Transactional FSL (26) datasets.
 * Hand-only 63 features. Validation accuracy: 44%.
 */
export const INFERENCE_LABELS: string[] = [
  "A",
  "AGAIN",
  "B",
  "C",
  "CARD",
  "CASH",
  "COIN",
  "Confirm",
  "D",
  "DISCOUNT",
  "E",
  "EIGHT",
  "F",
  "FIVE",
  "FOUR",
  "G",
  "H",
  "HELLO",
  "HOW_MANY",
  "HOW_MUCH",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "NINE",
  "NO",
  "O",
  "ONE",
  "P",
  "PLEASE",
  "PROBLEM",
  "Q",
  "R",
  "RECEIPT",
  "S",
  "SEVEN",
  "SIX",
  "Space",
  "Speak",
  "T",
  "TEN",
  "THANK_YOU",
  "THREE",
  "TWO",
  "U",
  "V",
  "W",
  "WAIT",
  "WELCOME",
  "X",
  "Y",
  "YES",
];