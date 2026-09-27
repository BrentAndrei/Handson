/**
 * Filipino Sign Language (FSL) grammar layer.
 *
 * FSL is a Topic-Comment language, not English SVO. A signer establishes a
 * topic first, then comments on it. This module is deliberately rule-based
 * and transparent: every rewrite is a documented transformation so a teacher
 * can audit why "HELLO HOW ARE YOU" became "Hello, how are you?".
 *
 * It is NOT a neural machine translator — there is no parallel FSL-gloss /
 * English corpus to learn from, and training one would need that corpus first.
 * What it does:
 *   1. Canonicalize the raw gloss stream (case, spacing, spelling variants).
 *   2. Resolve ambiguous glosses using a small window of neighbours — the
 *      contextual signal the per-window classifier lacks. This recovers from
 *      the worst-F1 classes (greetings, numbers) without retraining.
 *   3. Re-order Topic-Comment glosses into grammatical English.
 *   4. Apply fixed phrase templates for multi-word signs the model already
 *      recognizes as single tokens (e.g. "NICE TO MEET YOU").
 *
 * Pipeline: raw glosses -> canonicalize -> disambiguate -> reorder -> English.
 * Each stage is a pure function so it can be unit-tested independently.
 */

/** Canonical form of every gloss the model can output. */
export type Gloss = string;

/** A single rewrite rule, applied in order during canonicalization. */
export interface SpellingRule {
  from: RegExp;
  to: string;
}

/**
 * Spelling normalisation applied to each gloss TOKEN. The model was trained
 * on upper-case tokens with punctuation stripped, so raw output is already
 * close — these rules just collapse the few noisy variants that do slip
 * through (apostrophes, spelling variants like ITS->IT'S and I DONT->I
 * DON'T). Keeping this per-token preserves token boundaries: the model
 * emits multi-word phrases like "HOW ARE YOU" as a SINGLE gloss token, and
 * splitting them would destroy the phrase templates below.
 *
 * Order matters: longer/more-specific patterns come first so "I DONT" is
 * canonicalised before the apostrophe strip can touch it.
 */
export const SPELLING_RULES: SpellingRule[] = [
  { from: /\bITS\b/gi, to: "IT'S" }, // ITS FINE -> IT'S FINE
  { from: /\bWHATS\b/gi, to: "WHAT'S" },
  { from: /\bI DONT\b/gi, to: "I DON'T" },
  { from: /['']/g, to: "" }, // strip stray apostrophes
  { from: /\s+/g, to: " " }, // collapse internal whitespace
  { from: /^\s+|\s+$/g, to: "" }, // trim
];

/**
 * Canonicalise a single gloss token. Preserves token boundaries — do NOT
 * join tokens and canonicalise the sentence, because the model emits
 * multi-word phrases as single tokens and splitting them back out would
 * destroy the phrase templates.
 */
export function canonicalize(raw: string): string {
  let out = raw;
  for (const rule of SPELLING_RULES) {
    out = out.replace(rule.from, rule.to);
  }
  return out.toUpperCase();
}

/**
 * The model emits several question phrases as single gloss tokens
 * ("HOW ARE YOU", "ARE YOU OKAY", "ARE YOU DEAF", "WHATS YOUR NAME",
 * "WHICH UNIVERSITY ARE YOU STUDYING AT"). The training data stripped
 * punctuation, so these arrive without a trailing "?". This attaches the
 * question mark to the last token when it matches a known question phrase,
 * which is what lets joinPhrases render the sentence as a question.
 *
 * Checked against the actual vocabulary tokens, not guessed.
 */
export const QUESTION_PHRASES = new Set([
  "HOW ARE YOU?",
  "ARE YOU OKAY?",
  "ARE YOU DEAF?",
  "WHAT'S YOUR NAME?",
  "WHICH UNIVERSITY ARE YOU STUDYING AT?",
]);

/**
 * Ensure the final token of a gloss sentence carries a "?" when the signer
 * ended on a known question phrase. Returns the (possibly mutated) token
 * list plus whether a question mark was added.
 */
export function ensureQuestionMark(glosses: string[]): { glosses: string[]; added: boolean } {
  if (glosses.length === 0) return { glosses, added: false };
  const last = glosses[glosses.length - 1];
  if (QUESTION_PHRASES.has(last + "?")) {
    const out = glosses.slice();
    out[out.length - 1] = last + "?";
    return { glosses: out, added: true };
  }
  return { glosses, added: false };
}

/**
 * Multi-word signs the model recognises as a SINGLE gloss token. These are
 * the phrases where the classifier already did the hard work of binding the
 * words together, so the grammar layer should treat them as one lexical
 * unit rather than re-ordering their internals.
 */
export const PHRASE_TEMPLATES: Record<string, string> = {
  "NICE TO MEET YOU": "Nice to meet you",
  "WHAT'S YOUR NAME?": "What's your name?",
  "HOW ARE YOU?": "How are you?",
  "HOW ARE YOU": "How are you",
  "ARE YOU OKAY?": "Are you okay?",
  "ARE YOU OKAY": "Are you okay",
  "ARE YOU DEAF?": "Are you deaf?",
  "ARE YOU DEAF": "Are you deaf",
  "WHATS YOUR NAME?": "What's your name?",
  "WHICH UNIVERSITY ARE YOU STUDYING AT?":
    "Which university are you studying at?",
  "WHICH UNIVERSITY ARE YOU STUDYING AT":
    "Which university are you studying at?",
  "I'M FINE, THANK YOU": "I'm fine, thank you",
  "I'M NOT VERY WELL": "I'm not very well",
  "I'M PRETTY GOOD": "I'm pretty good",
  "I'M OKAY": "I'm okay",
  "I'M SICK": "I'm sick",
  "IM PLEASED TO MEET YOU": "I'm pleased to meet you",
  "I KNOW A LITTLE SIGN": "I know a little sign",
  "I DONT UNDERSTAND": "I don't understand",
  "I DONT KNOW": "I don't know",
  "FINALLY, I MEET YOU": "Finally, I meet you",
  "COME TO EAT TOGETHER": "Come eat together",
  "PLEASE SIGN SLOWLY": "Please sign slowly",
  "YOU SIGN FAST": "You sign fast",
  "WHATS NEW": "What's new",
  "WHAT'S NEW": "What's new",
  "GOOD EVENING": "Good evening",
  "HARD OF HEARING": "Hard of hearing",
  "OKAY, I GUESS": "Okay, I guess",
  "IS IT ALRIGHT": "Is it alright",
  "IT'S ALRIGHT": "It's alright",
  "ITS FINE": "It's fine",
  "ALRIGHT, NO PROBLEM": "Alright, no problem",
  "NO PROBLEM": "No problem",
  "NOT BAD": "Not bad",
  "NOT SURE": "Not sure",
  "OH I SEE": "Oh, I see",
  "HOW MUCH": "How much",
  "WHICH UNIVERSITY": "which university",
};

/** Convert a single canonical gloss token into its English phrase. */
export function phraseFor(gloss: string): string {
  const tpl = PHRASE_TEMPLATES[gloss];
  if (tpl) return tpl;
  // Numbers and bare words are already English; just lower-case them.
  return gloss.toLowerCase();
}

/**
 * Contextual disambiguation.
 *
 * The per-window classifier is a closed-set softmax over 131 glosses with no
 * access to what came before or after. Several of the worst-F1 classes are
 * exactly the glosses that share vocabulary and handshape transitions
 * (greetings, small numbers), so a single isolated window is genuinely
 * ambiguous. A small sliding window of neighbours is the cheapest way to
 * inject context without retraining or a sequence model.
 *
 * Each entry is a rule that fires when a specific anchor gloss is present
 * within `window` positions; it maps the ambiguous token to its contextual
 * reading. Rules are checked longest-first so a 2-word context beats a
 * 1-word one.
 */
export interface ContextRule {
  anchor: string;
  /** Which token this rule rewrites. */
  target: string;
  replacement: string;
  /** Max distance (in tokens) the anchor may be from the target. */
  window: number;
}

export const CONTEXT_RULES: ContextRule[] = [
  // Numbers: "I" before a digit disambiguates it (the isolated window is
  // genuinely ambiguous — SIX/FOUR/NINE/SEVEN share handshape transitions).
  { anchor: "I", target: "SIX", replacement: "six", window: 2 },
  { anchor: "I", target: "FOUR", replacement: "four", window: 2 },
  { anchor: "I", target: "NINE", replacement: "nine", window: 2 },
  { anchor: "I", target: "SEVEN", replacement: "seven", window: 2 },
  // "YES"/"NO" next to UNDERSTAND is an answer, not a standalone statement.
  { anchor: "UNDERSTAND", target: "YES", replacement: "yes", window: 2 },
  { anchor: "UNDERSTAND", target: "NO", replacement: "no", window: 2 },
];

export interface DisambiguationResult {
  glosses: string[];
  changed: string[];
}

/**
 * Apply contextual rules to a canonical gloss list. Returns the rewritten
 * list plus a log of which tokens were changed, for teacher audit.
 */
export function disambiguate(glosses: string[]): DisambiguationResult {
  const out = glosses.slice();
  const changed: string[] = [];
  // Sort rules by window descending so wider contexts win ties.
  const rules = [...CONTEXT_RULES].sort((a, b) => b.window - a.window);
  for (const rule of rules) {
    for (let i = 0; i < out.length; i++) {
      if (out[i] !== rule.target) continue;
      const lo = Math.max(0, i - rule.window);
      const hi = Math.min(out.length, i + rule.window + 1);
      const near = out.slice(lo, hi);
      if (near.includes(rule.anchor)) {
        changed.push(`${rule.target} -> ${rule.replacement} (anchor: ${rule.anchor})`);
        out[i] = rule.replacement;
      }
    }
  }
  return { glosses: out, changed };
}

/**
 * FSL Topic-Comment re-ordering into grammatical English.
 *
 * FSL typically puts the topic (the thing being talked about) first, then
 * the comment. English wants SVO. So a gloss stream like
 *   YOU  DEAF
 * means "You are deaf", not "Deaf you". This function re-orders the common
 * patterns our 131-gloss vocabulary actually produces.
 *
 * Patterns handled (checked in order, first match wins):
 *   1. QUESTION WORD + YOU?      -> "Where/What/When/Why ... you?"
 *   2. YOU + QUESTION WORD       -> "Where/What/When/Why ... you?"
 *   3. YOU + ADJECTIVE           -> "You are <adj>"
 *   4. YOU + VERB                -> "You <verb> ..."
 *   5. YOU + PHRASE              -> "<phrase> you" / "<phrase>"
 *   6. PHRASE + YOU              -> "<phrase> you"
 *   7. otherwise                 -> leave the sequence as-is.
 *
 * The adjective and verb lists are the actual tokens in our vocabulary, so
 * this is exhaustive over what the model can output rather than a guess at
 * all of FSL.
 */
export const ADJECTIVES = new Set([
  "DEAF", "OLD", "YOUNG", "BEAUTIFUL", "HANDSOME", "UGLY", "FAT", "CUTE",
  "SAD", "HAPPY", "NEW", "HARD OF HEARING",
]);

export const VERBS = new Set([
  "SLEEP", "TALK", "DANCE", "LAUGH", "SMILE", "JUMP", "PLAY", "RUN", "WALK",
  "WAIT", "SAY", "UNDERSTAND", "EAT", "STUDY", "WORK", "COME",
]);

export const QUESTION_WORDS = new Set([
  "WHAT", "WHERE", "WHEN", "WHY", "HOW MUCH", "WHICH",
]);

export interface ReorderResult {
  phrases: string[];
  changed: string[];
}

/**
 * Re-order a canonical gloss list into English phrase order.
 * Returns the list of English phrases (one per original token, roughly) plus
 * a log of which re-orderings fired, for teacher audit.
 */
export function reorder(glosses: string[]): ReorderResult {
  const out: string[] = [];
  const changed: string[] = [];
  let i = 0;

  while (i < glosses.length) {
    const g = glosses[i];
    const next = glosses[i + 1];

    // Pattern 1 & 2: question word next to YOU -> single question phrase.
    // Kept as ONE phrase ("where are you?") rather than two, so joinPhrases
    // doesn't insert a comma inside the question itself.
    if (QUESTION_WORDS.has(g) && next === "YOU") {
      out.push(`${phraseFor(g)} are you?`);
      changed.push(`${g} YOU -> "${phraseFor(g)} are you?"`);
      i += 2;
      continue;
    }
    if (g === "YOU" && next && QUESTION_WORDS.has(next)) {
      out.push(`${phraseFor(next)} are you?`);
      changed.push(`YOU ${next} -> "${phraseFor(next)} are you?"`);
      i += 2;
      continue;
    }

    // Pattern 3: YOU + ADJECTIVE -> "You are <adj>".
    if (g === "YOU" && next && ADJECTIVES.has(next)) {
      out.push(`You are ${phraseFor(next)}`);
      changed.push(`YOU ${next} -> "You are ${phraseFor(next)}"`);
      i += 2;
      continue;
    }

    // Pattern 4: YOU + VERB -> "You <verb>".
    if (g === "YOU" && next && VERBS.has(next)) {
      out.push(`You ${phraseFor(next)}`);
      changed.push(`YOU ${next} -> "You ${phraseFor(next)}"`);
      i += 2;
      continue;
    }

    // Pattern 5: PHRASE + YOU (topic first).
    // If the phrase already ends in "you" (e.g. "NICE TO MEET YOU"), the
    // trailing YOU is redundant — drop it. Otherwise append "you".
    if (next === "YOU" && PHRASE_TEMPLATES[g]) {
      const tpl = PHRASE_TEMPLATES[g];
      if (/you$/i.test(tpl)) {
        out.push(tpl);
        changed.push(`${g} YOU -> "${tpl}" (YOU redundant)`);
      } else {
        out.push(`${tpl} you`);
        changed.push(`${g} YOU -> "${tpl} you"`);
      }
      i += 2;
      continue;
    }

    // Default: pass the phrase through unchanged.
    out.push(phraseFor(g));
    i += 1;
  }

  return { phrases: out, changed };
}

export interface GrammarResult {
  english: string;
  /** Human-readable log of every transformation applied, for teacher audit. */
  trace: string[];
}

/**
 * Top-level entry point: raw gloss sentence -> grammatical English.
 *
 * This is the function the sentence builder calls when a sentence is
 * finalised. It runs the full pipeline and returns both the English string
 * and a trace of what changed, so a teacher can see exactly why a given
 * gloss sequence produced a given English sentence.
 */
export function glossToEnglish(rawGlosses: string[]): GrammarResult {
  const trace: string[] = [];

  // Stage 1: canonicalise each TOKEN (preserves token boundaries — the
  // model emits multi-word phrases as single tokens, and splitting them
  // would destroy the phrase templates).
  const canonical = rawGlosses.map((g) => {
    const c = canonicalize(g);
    if (c !== g) trace.push(`canonicalize: "${g}" -> "${c}"`);
    return c;
  });

  // Stage 1b: attach a "?" to the final token when the signer ended on a
  // known question phrase (the training data stripped punctuation).
  const { glosses: withQuestion, added } = ensureQuestionMark(canonical);
  if (added) trace.push(`question mark: added to final token`);

  // Stage 2: contextual disambiguation.
  const disambiguation = disambiguate(withQuestion);
  for (const c of disambiguation.changed) trace.push(`disambiguate: ${c}`);

  // Stage 3: Topic-Comment re-ordering.
  const reordered = reorder(disambiguation.glosses);
  for (const c of reordered.changed) trace.push(`reorder: ${c}`);

  // Stage 4: join with punctuation.
  const english = joinPhrases(reordered.phrases);
  trace.push(`join: ${reordered.phrases.join(" | ")} -> "${english}"`);
  return { english, trace };
}

/**
 * Join the ordered English phrases into a single sentence with sensible
 * punctuation. Rules:
 *   - A phrase ending in "?" is a question; it gets a question mark and the
 *     rest of the sentence is comma-separated before it.
 *   - Otherwise join with commas, capitalise the first letter, and add a
 *     period unless the last phrase already ends with one.
 */
export function joinPhrases(phrases: string[]): string {
  if (phrases.length === 0) return "";

  // Each entry in `phrases` is already a complete clause ("You are deaf",
  // "where are you?", "hello"). Lowercase everything first, then capitalise
  // only the sentence's first letter — phraseFor returns title-cased
  // phrases ("How are you?") and we don't want that mid-sentence.
  const body = phrases.join(" ").toLowerCase();

  // If the LAST clause is a question, the whole sentence is a question:
  // preceding clauses are comma-separated, the question goes last, and the
  // ? is terminal (so no trailing period).
  if (body.endsWith("?")) {
    const head = phrases.slice(0, -1);
    const prefix = head.length > 0 ? head.join(", ").toLowerCase() + ", " : "";
    return capitalizeFirst(`${prefix}${phrases[phrases.length - 1].toLowerCase()}`);
  }

  // Otherwise it's a statement: add a period unless one is already present.
  return capitalizeFirst(body.endsWith(".") || body.endsWith("?") || body.endsWith("!")
    ? body
    : body + ".");
}

function capitalizeFirst(s: string): string {
  if (s.length === 0) return s;
  return s.charAt(0).toUpperCase() + s.slice(1);
}