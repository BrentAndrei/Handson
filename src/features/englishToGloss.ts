/**
 * English → FSL gloss converter (Diagram 2, reverse of grammarParser).
 *
 * FSL is a Topic-Comment language, not English SVO. A signer establishes a
 * topic first, then comments on it. This module takes English text and
 * produces a structured gloss sequence in Topic-Comment order.
 *
 * It is NOT a neural machine translator — there is no parallel English /
 * FSL-gloss corpus. What it does:
 *   1. Detects sentence type (question, command, statement).
 *   2. Looks up English phrases and words in a starter dictionary
 *      drawn from the 131-label model vocabulary.
 *   3. Re-orders English SVO into FSL Topic-Comment for a few common
 *      patterns that the model's labels cover.
 *   4. Flags unknown words so a teacher knows what the vocabulary doesn't handle.
 *
 * Pipeline: English → sentence type → phrase lookup → word lookup
 * → structural reordering → gloss sequence. Each stage is a pure function.
 */

/** A single FSL gloss token with its English gloss and structural role. */
export interface GlossToken {
  /** The FSL gloss label (e.g. "HELLO", "YOU", "DEAF"). */
  gloss: string;
  /** Rough English meaning of this gloss token. */
  meaning: string;
  /** Structural role in the Topic-Comment structure. */
  role: "topic" | "comment" | "unknown";
}

export interface EnglishToGlossResult {
  /** Structured gloss sequence in Topic-Comment order. */
  glosses: GlossToken[];
  /** Human-readable log of every transformation, for teacher audit. */
  trace: string[];
  /** Sentence type detected from the input. */
  sentenceType: "question" | "command" | "statement";
}

// ---------------------------------------------------------------------------
// Starter dictionary — English → FSL gloss
// Built by inverting PHRASE_TEMPLATES from grammarParser.ts and adding the
// single-word labels from the 131-label vocabulary that have direct English
// equivalents. This is a STARTER vocabulary; unknown words are flagged, not
// silently dropped.
// ---------------------------------------------------------------------------

/** English phrase/word (lowercase, no punctuation) → FSL gloss token. */
const ENGLISH_TO_GLOSS: Record<string, string> = {
  // Greetings & common phrases (from PHRASE_TEMPLATES, inverted)
  "nice to meet you": "NICE TO MEET YOU",
  "what's your name": "WHAT'S YOUR NAME",
  "whats your name": "WHAT'S YOUR NAME",
  "what's your name?": "WHAT'S YOUR NAME",
  "how are you": "HOW ARE YOU",
  "how are you?": "HOW ARE YOU",
  "are you okay": "ARE YOU OKAY",
  "are you okay?": "ARE YOU OKAY",
  "are you deaf": "ARE YOU DEAF",
  "are you deaf?": "ARE YOU DEAF",
  "i'm fine, thank you": "I'M FINE, THANK YOU",
  "i'm fine thank you": "I'M FINE, THANK YOU",
  "i'm not very well": "I'M NOT VERY WELL",
  "i'm pretty good": "I'M PRETTY GOOD",
  "i'm okay": "I'M OKAY",
  "i'm sick": "I'M SICK",
  "i'm pleased to meet you": "IM PLEASED TO MEET YOU",
  "i know a little sign": "I KNOW A LITTLE SIGN",
  "i don't understand": "I DONT UNDERSTAND",
  "i don't know": "I DONT KNOW",
  "finally i meet you": "FINALLY, I MEET YOU",
  "come to eat together": "COME TO EAT TOGETHER",
  "please sign slowly": "PLEASE SIGN SLOWLY",
  "you sign fast": "YOU SIGN FAST",
  "what's new": "WHAT'S NEW",
  "good evening": "GOOD EVENING",
  "hard of hearing": "HARD OF HEARING",
  "okay i guess": "OKAY, I GUESS",
  "is it alright": "IS IT ALRIGHT",
  "it's alright": "IT'S ALRIGHT",
  "it's fine": "ITS FINE",
  "alright, no problem": "ALRIGHT, NO PROBLEM",
  "no problem": "NO PROBLEM",
  "not bad": "NOT BAD",
  "not sure": "NOT SURE",
  "oh i see": "OH I SEE",
  "how much": "HOW MUCH",
  "which university": "WHICH UNIVERSITY",
  "hello": "HELLO",
  "hi": "HELLO",
  "goodbye": "GOODBYE",
  "bye": "GOODBYE",
  "thank you": "THANK YOU",
  "thanks": "THANK YOU",
  "sorry": "SORRY",
  "yes": "YES",
  "no": "NO",
  "please": "PLEASE",
  "welcome": "WELCOME",
  "sure": "SURE",
  "okay": "OKAY",
  "today": "DAY",
  "tomorrow": "TOMORROW",
  "now": "NOW",
  "here": "HERE",
  "there": "THERE",
  "this": "THIS",
  "that": "THAT",
  "what": "WHAT",
  "who": "WHO",
  "where": "WHERE",
  "when": "WHEN",
  "why": "WHY",
  "how": "HOW",
  "which": "WHICH",
  "is": "IS",
  "are": "ARE",
  "am": "AM",
  "was": "WAS",
  "were": "WERE",
  "will": "WILL",
  "can": "CAN",
  "could": "COULD",
  "would": "WOULD",
  "should": "SHOULD",
  "may": "MAY",
  "might": "MAY",
  "must": "MUST",
  "do": "DO",
  "does": "DOES",
  "did": "DID",
  "have": "HAVE",
  "has": "HAS",
  "had": "HAD",
  "been": "BEEN",
  "being": "BEING",
  "i": "I",
  "you": "YOU",
  "he": "HE",
  "she": "HE/SHE",
  "it": "IT",
  "we": "WE",
  "they": "THEY",
  "me": "ME",
  "him": "HIM",
  "her": "HER",
  "us": "US",
  "them": "THEM",
  "my": "MY",
  "your": "YOUR",
  "his": "HIS",
  "its": "ITS",
  "our": "OUR",
  "their": "THEIR",
  "mine": "MINE",
  "yours": "YOURS",
  "hers": "HERS",
  "ours": "OURS",
  "theirs": "THEIRS",
  // Adjectives (Topic-Comment: ADJ YOU → "You are ADJ")
  "happy": "HAPPY",
  "sad": "SAD",
  "deaf": "DEAF",
  "old": "OLD",
  "young": "YOUNG",
  "beautiful": "BEAUTIFUL",
  "handsome": "HANDSOME",
  "ugly": "UGLY",
  "fat": "FAT",
  "cute": "CUTE",
  "new": "NEW",
  // Verbs (Topic-Comment: VERB YOU → "You VERB")
  "sleep": "SLEEP",
  "talk": "TALK",
  "dance": "DANCE",
  "laugh": "LAUGH",
  "smile": "SMILE",
  "jump": "JUMP",
  "play": "PLAY",
  "run": "RUN",
  "walk": "WALK",
  "wait": "WAIT",
  "say": "SAY",
  "understand": "UNDERSTAND",
  "know": "KNOW",
  "meet": "MEET",
  "eat": "EAT",
  "sign": "SIGN",
  "study": "STUDY",
  "work": "WORK",
  "come": "COME",
  "go": "GO",
  "see": "SEE",
  "hear": "HEAR",
  "read": "READ",
  "write": "WRITE",
  "learn": "LEARN",
  "teach": "TEACH",
  "live": "LIVE",
  "like": "LIKE",
  "love": "LOVE",
  "want": "WANT",
  "need": "NEED",
  "think": "THINK",
  "feel": "FEEL",
  "make": "MAKE",
  "take": "TAKE",
  "give": "GIVE",
  "tell": "TELL",
  "ask": "ASK",
  "help": "HELP",
  "try": "TRY",
  "use": "USE",
  "find": "FIND",
  "keep": "KEEP",
  "start": "START",
  "stop": "STOP",
  "show": "SHOW",
  "hear deaf": "DEAF",
  // Days & numbers (direct equivalents)
  "monday": "MONDAY",
  "tuesday": "TUESDAY",
  "wednesday": "WEDNESDAY",
  "thursday": "THURSDAY",
  "friday": "FRIDAY",
  "saturday": "SATURDAY",
  "sunday": "SUNDAY",
  "week": "WEEK",
  "one": "ONE",
  "two": "TWO",
  "three": "THREE",
  "four": "FOUR",
  "five": "FIVE",
  "six": "SIX",
  "seven": "SEVEN",
  "eight": "EIGHT",
  "nine": "NINE",
  "ten": "TEN",
  "eleven": "ELEVEN",
  "twelve": "TWELVE",
  "thirteen": "THIRTEEN",
  "fourteen": "FOURTEEN",
  "fifteen": "FIFTEEN",
  "sixteen": "SIXTEEN",
  "seventeen": "SEVENTEEN",
  "eighteen": "EIGHTEEN",
  "nineteen": "NINETEEN",
  "twenty": "TWENTY",
  // ---------------------------------------------------------------------------
  // Additional vocabulary from INFERENCE_LABELS
  // ---------------------------------------------------------------------------
  "again": "AGAIN",
  "because": "BECAUSE",
  "cool": "COOL",
  "from": "FROM",
  "grandchild": "GRANDCHILD",
  "hearing": "HEARING",
  "his/her": "HIS/HER",
  "introduction": "INTRODUCTION",
  "problem": "PROBLEM",
  "responses": "RESPONSES",
  "somewhere": "SOMEWHERE",
  "thank": "THANK",
  "then": "THEN",
  "than": "THAN",
  "thin": "THIN",
  "too": "TOO",
  "wow": "WOW",
  "i understand": "I UNDERSTAND",
  "nowhere": "NOWHERE",
  "whats new": "WHAT'S NEW",
};

// ---------------------------------------------------------------------------
// Sentence type detection
// ---------------------------------------------------------------------------

export type SentenceType = "question" | "command" | "statement";

/**
 * Detect sentence type from punctuation and structure.
 * Questions end with "?" (or contain question words at the start).
 * Commands typically start with an imperative verb.
 * Everything else is a statement.
 */
export function detectSentenceType(text: string): SentenceType {
  const trimmed = text.trim();
  if (trimmed.endsWith("?")) return "question";
  if (trimmed.endsWith("!")) return "command";
  const firstWord = trimmed.split(/\s+/)[0]?.toUpperCase();
  // Common imperative starters
  const COMMAND_STARTS = new Set([
    "PLEASE", "SIGN", "SHOW", "TELL", "GIVE", "TRY", "HELP", "START",
    "STOP", "LOOK", "LISTEN", "WATCH", "WRITE", "READ", "SAY", "TEACH",
    "LEARN", "COME", "GO", "MAKE", "DO", "BE", "HAVE", "WANT", "NEED",
  ]);
  if (firstWord && COMMAND_STARTS.has(firstWord)) return "command";
  return "statement";
}

// ---------------------------------------------------------------------------
// Tokenization & lookup
// ---------------------------------------------------------------------------

/**
 * Split English text into tokens, stripping punctuation and normalizing.
 * Preserves multi-word phrases by checking the dictionary first.
 */
function tokenizeEnglish(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s']/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 0);
}

/**
 * Try to match the longest possible English phrase from the dictionary,
 * starting at position `start` in the token array.
 * Returns the matched phrase's gloss or null.
 */
function tryPhraseMatch(
  tokens: string[],
  start: number,
  maxPhraseLen: number
): { gloss: string; length: number } | null {
  for (let len = maxPhraseLen; len >= 1; len--) {
    if (start + len > tokens.length) continue;
    const phrase = tokens.slice(start, start + len).join(" ");
    const gloss = ENGLISH_TO_GLOSS[phrase];
    if (gloss) return { gloss, length: len };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Structural reordering — English SVO → FSL Topic-Comment
// ---------------------------------------------------------------------------

const QUESTION_WORDS = new Set(["WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHICH"]);

const ADJECTIVES = new Set([
  "HAPPY", "SAD", "DEAF", "OLD", "YOUNG", "BEAUTIFUL", "HANDSOME",
  "UGLY", "FAT", "CUTE", "NEW", "HARD OF HEARING",
]);

const VERBS = new Set([
  "SLEEP", "TALK", "DANCE", "LAUGH", "SMILE", "JUMP", "PLAY", "RUN",
  "WALK", "WAIT", "SAY", "UNDERSTAND", "KNOW", "MEET", "EAT", "SIGN",
  "STUDY", "WORK", "COME", "GO", "SEE", "HEAR", "READ", "WRITE", "LEARN",
  "TEACH", "LIVE", "LIKE", "LOVE", "WANT", "NEED", "THINK", "FEEL",
  "MAKE", "TAKE", "GIVE", "TELL", "ASK", "HELP", "TRY", "USE", "FIND",
  "KEEP", "START", "STOP", "SHOW",
]);

const TIME_WORDS = new Set([
  "DAY", "TODAY", "TONIGHT", "TOMORROW", "YESTERDAY",
  "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
  "WEEK", "WEEKEND", "MORNING", "AFTERNOON", "EVENING", "NIGHT",
]);

const POSSESSIVES = new Set([
  "MY", "YOUR", "HIS", "HER", "ITS", "OUR", "YOUR", "THEIR",
  "MINE", "YOURS", "HERS", "OURS", "THEIRS",
]);

/**
 * Apply Topic-Comment structural reordering for patterns the 131-label
 * vocabulary covers. This is the INVERSE of grammarParser.ts's reorder()
 * patterns.
 *
 * Patterns (English → FSL), checked in order, first match wins:
 *   0. "You/I/He/She are/is X" → "X YOU/..." (BE verb dropped, adjective/verb is topic)
 *   A. "Time + X" → "X Time" (time is topic, placed last in FSL)
 *   B. "My/Your + X" → "X My/Your" (possessed thing is topic, possessor is comment)
 *   1. "What/Where/When/Why ... you?" → "WHAT YOU?" (WH-word is topic)
 *   2. "You are X" → "X YOU" (adjective is topic, you is comment)
 *   3. "You X" → "X YOU" (verb is topic, you is comment)
 */
function reorderToTopicComment(
  glosses: GlossToken[]
): { glosses: GlossToken[]; trace: string[] } {
  const out: GlossToken[] = [];
  const trace: string[] = [];
  let i = 0;

  while (i < glosses.length) {
    const g = glosses[i];
    const next = glosses[i + 1];
    const third = glosses[i + 2];

    // Pattern 0: "You/I/He/She are/is X" → "X YOU/..." (3-word: BE verb is dropped)
    if (
      (g.gloss === "YOU" || g.gloss === "I" || g.gloss === "HE/SHE" || g.gloss === "WE") &&
      next && (next.gloss === "ARE" || next.gloss === "IS" || next.gloss === "AM") &&
      third && (ADJECTIVES.has(third.gloss) || VERBS.has(third.gloss))
    ) {
      out.push({ ...third, role: "topic" });
      out.push({ ...g, role: "comment" });
      trace.push(
        `reorder: ${g.gloss} ${next.gloss} ${third.gloss} → ${third.gloss} is topic, ${g.gloss} is comment (Topic-Comment order)`
      );
      i += 3;
      continue;
    }

    // Pattern A: "Time + X" → "X Time" (time is topic in FSL, placed last)
    if (
      TIME_WORDS.has(g.gloss) &&
      next && !TIME_WORDS.has(next.gloss)
    ) {
      out.push({ ...next, role: "topic" });
      out.push({ ...g, role: "comment" });
      trace.push(`reorder: ${g.gloss} ${next.gloss} → ${next.gloss} is topic, ${g.gloss} is comment (time is topic, placed last in FSL)`);
      i += 2;
      continue;
    }

    // Pattern B: "My/Your + X" → "X My/Your" (possessed thing is topic)
    if (
      POSSESSIVES.has(g.gloss) &&
      next && !POSSESSIVES.has(next.gloss) && !VERBS.has(next.gloss) && !ADJECTIVES.has(next.gloss) && !QUESTION_WORDS.has(next.gloss)
    ) {
      out.push({ ...next, role: "topic" });
      out.push({ ...g, role: "comment" });
      trace.push(`reorder: ${g.gloss} ${next.gloss} → ${next.gloss} is topic, ${g.gloss} is comment (possessed thing is topic)`);
      i += 2;
      continue;
    }

    // Pattern 1: Question word followed by YOU/pronoun → WH is topic
    if (
      QUESTION_WORDS.has(g.gloss) &&
      next &&
      (next.gloss === "YOU" || next.gloss === "HE/SHE" || next.gloss === "WE" || next.gloss === "I")
    ) {
      out.push({ ...g, role: "topic" });
      out.push({ ...next, role: "comment" });
      trace.push(`reorder: ${g.gloss} ${next.gloss} → WH-word is topic, pronoun is comment`);
      i += 2;
      continue;
    }

    // Pattern 2 & 3: YOU/pronoun followed by adjective or verb → ADJ/VERB is topic
    if (
      (g.gloss === "YOU" || g.gloss === "HE/SHE" || g.gloss === "WE" || g.gloss === "I") &&
      next &&
      (ADJECTIVES.has(next.gloss) || VERBS.has(next.gloss))
    ) {
      out.push({ ...next, role: "topic" });
      out.push({ ...g, role: "comment" });
      trace.push(
        `reorder: ${g.gloss} ${next.gloss} → ${next.gloss} is topic, ${g.gloss} is comment (Topic-Comment order)`
      );
      i += 2;
      continue;
    }

    // Default: pass through unchanged
    out.push({ ...g, role: "unknown" });
    i++;
  }

  return { glosses: out, trace };
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/**
 * Convert English text to a structured FSL gloss sequence in Topic-Comment order.
 *
 * @param text English sentence (typed by a hearing user)
 * @returns Gloss sequence, trace of transformations, and sentence type
 */
export function englishToGloss(text: string): EnglishToGlossResult {
  const trace: string[] = [];
  const sentenceType = detectSentenceType(text);
  trace.push(`sentence type: ${sentenceType}`);

  // Stage 1: Tokenize
  const tokens = tokenizeEnglish(text);
  trace.push(`tokens: [${tokens.join(", ")}]`);

  // Stage 2: Phrase-level lookup (longest match first)
  const glosses: GlossToken[] = [];
  let i = 0;
  while (i < tokens.length) {
    const match = tryPhraseMatch(tokens, i, 5);
    if (match) {
      glosses.push({ gloss: match.gloss, meaning: match.gloss, role: "unknown" });
      trace.push(`phrase: "${tokens.slice(i, i + match.length).join(" ")}" → "${match.gloss}"`);
      i += match.length;
    } else {
      const word = tokens[i].toUpperCase();
      const known = ENGLISH_TO_GLOSS[tokens[i]];
      if (known) {
        glosses.push({ gloss: known, meaning: known, role: "unknown" });
        trace.push(`word: "${tokens[i]}" → "${known}"`);
      } else {
        glosses.push({ gloss: word, meaning: word, role: "unknown" });
        trace.push(`unknown: "${tokens[i]}" — not in starter vocabulary`);
      }
      i++;
    }
  }

  // Stage 3: Topic-Comment reordering for known patterns
  const reordered = reorderToTopicComment(glosses);
  trace.push(...reordered.trace);

  // Stage 4: Annotate question marks for question sentence type
  if (sentenceType === "question") {
    for (const g of reordered.glosses) {
      if (g.gloss.endsWith("?") || QUESTION_WORDS.has(g.gloss)) {
        if (!g.gloss.endsWith("?")) {
          g.gloss = g.gloss + "?";
        }
      }
    }
  }

  return {
    glosses: reordered.glosses,
    trace,
    sentenceType,
  };
}
