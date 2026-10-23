/**
 * Cheap-turn disposition (GDD §9.10.2: "the model proposes; the engine disposes").
 *
 * When the dialogue role points at a non-Claude model (MIND_PROVIDER=openai, e.g. Ollama's gemma4:e2b),
 * the small model writes only the resident's lines (cheapDialogue.ts); this module disposes the rest:
 * the choices the Fellow can pick — ranked from an authored candidate pool by embedding similarity, or
 * deterministically when no embedding model is configured — and the turn's effects (regard and mood from
 * a nearest-sentiment-anchor, wrap_up from the turn's focus). Everything here is pure: the embedder is
 * injected, so the module runs under vitest with no network.
 */
import { DialogueTurnOutput, toPortraitExpression, type Intent } from "@monde/shared";
import type { TurnFocus } from "./promptBuilder.js";
export type { Embedder } from "./openaiClient.js";
import type { Embedder } from "./openaiClient.js";

/** A reply the Fellow could give, first person, in the Fellow's friendly, slightly dry voice (world bible
 * Output section). Choices are the player's words, so the pool stays resident-agnostic. */
export interface ChoiceCandidate {
  text: string;
  intent: Intent;
}

/**
 * The authored choice pool the scorer ranks. Deliberately plain — these are what the Fellow says back to
 * any resident's lines; the resident-specific voice lives in the lines the model writes.
 */
export const CHOICE_POOL: readonly ChoiceCandidate[] = [
  { text: "Tell me more.", intent: "curious" },
  { text: "How'd you come by that?", intent: "curious" },
  { text: "What would you make of it?", intent: "curious" },
  { text: "That means a lot, thank you.", intent: "kind" },
  { text: "You've been kind to me.", intent: "kind" },
  { text: "I'd like to help with that.", intent: "ask_favor" },
  { text: "Would you do me a favor?", intent: "ask_favor" },
  { text: "Now that, I can get behind.", intent: "joke" },
  { text: "Ha. Fair enough.", intent: "joke" },
  { text: "I mean it, and I won't ask again.", intent: "firm" },
  { text: "Let's keep it simple.", intent: "firm" },
  { text: "It's on my mind, if you'd help.", intent: "about_quest" },
  { text: "How about you? How's it going?", intent: "campaign" },
  { text: "I should get back to it.", intent: "leave" },
  { text: "I'd better not keep you.", intent: "leave" },
];

/** Sentiment anchors for the Fellow's words: how each lands, in Regard terms (GDD §9.10.2: most turns 0 or +1). */
export interface SentimentAnchor {
  text: string;
  delta: number;
}
export const SENTIMENT_ANCHORS: readonly SentimentAnchor[] = [
  { text: "A warm, friendly, cheerful remark", delta: 1 },
  { text: "A genuine compliment, a sincere thank-you", delta: 2 },
  { text: "An offer of help, a gift of food or drink", delta: 2 },
  { text: "A shared laugh, friendly teasing", delta: 1 },
  { text: "A plain question about the day's work or the weather", delta: 0 },
  { text: "A polite but dismissive remark", delta: -1 },
  { text: "A rude, pushy or nosy remark", delta: -2 },
  { text: "Mockery, cruelty, an insult", delta: -3 },
];

export const cosine = (a: readonly number[], b: readonly number[]): number => {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
};

/** FNV-1a over code points: a stable, dependency-free rotation seed. */
export const hashText = (s: string): number => {
  let h = 0x811c9dc5;
  for (const c of s) {
    h ^= c.codePointAt(0) ?? 0;
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
};

const pickByScore = (ranked: readonly { cand: ChoiceCandidate; score: number }[], focus: TurnFocus): ChoiceCandidate[] => {
  // A walk-up greeting (focus 'open') does not end at the door: no leave candidate on that turn.
  const skipLeave = focus === "open";
  const picked: ChoiceCandidate[] = [];
  const seen = new Set<Intent>();
  for (const { cand } of ranked) {
    if (picked.length >= 3) break;
    if (skipLeave && cand.intent === "leave") continue;
    if (seen.has(cand.intent)) continue;
    picked.push(cand);
    seen.add(cand.intent);
  }
  // A goodbye turn needs an out; a mid-conversation one doesn't.
  if (focus === "bye" && !seen.has("leave")) {
    const leave = ranked.find((r) => r.cand.intent === "leave");
    if (leave) {
      picked[picked.length - 1] = leave.cand;
      seen.delete(leave.cand.intent);
      seen.add("leave");
    }
  }
  return picked;
};

/**
 * The three replies to offer the Fellow. Ranked by embedding similarity to what was just said (the
 * Fellow's words and the resident's lines); without an embedder — or if it fails — a deterministic
 * rotation of the authored pool keeps the turn live.
 */
export async function pickChoices(opts: {
  query: string;
  focus: TurnFocus;
  embedder?: Embedder | null;
}): Promise<ChoiceCandidate[]> {
  const { query, focus, embedder } = opts;
  let ranked: { cand: ChoiceCandidate; score: number }[] | null = null;
  if (embedder) {
    try {
      const vecs = await embedder([query, ...CHOICE_POOL.map((c) => c.text)]);
      ranked = CHOICE_POOL.map((cand, i) => ({ cand, score: cosine(vecs[0] ?? [], vecs[i + 1] ?? []) })).sort((a, b) => b.score - a.score);
    } catch {
      ranked = null;
    }
  }
  if (!ranked) {
    const start = hashText(query) % CHOICE_POOL.length;
    ranked = CHOICE_POOL
      .map((cand, i) => ({ cand, score: (i - start + CHOICE_POOL.length) % CHOICE_POOL.length }))
      .sort((a, b) => a.score - b.score);
  }
  return pickByScore(ranked, focus);
}

/**
 * The portrait expression for a cheap line: a question shows "thinking", otherwise the resident's current
 * mood emotion mapped to the portrait set (persona.ts), else neutral. Deterministic; the small model doesn't
 * choose expressions (GDD §9.10.2: the engine disposes).
 */
export function lineExpression(text: string, emotion?: string): string {
  if (/\?\s*$/.test(text.trim())) return "thinking";
  if (emotion) {
    const mapped = toPortraitExpression(emotion);
    if (mapped) return mapped;
  }
  return "neutral";
}

export interface CheapDisposition {
  effects: DialogueTurnOutput["effects"];
  wrapUp: boolean;
  /** A short tag for recent_topics (effects.ts): the model's free-form tag, decided by the engine here. */
  topic: string;
}

/** The recent-topics tag for a cheap turn: fixed per focus, so the engine never invents news. */
export function topicTag(focus: TurnFocus): string {
  switch (focus) {
    case "open": return "greetings";
    case "lull": return "chat";
    case "bye": return "farewell";
    case "gift": return "a gift";
    case "quest": return "the journal";
    case "reply": return "chat";
  }
}

/**
 * The turn's effects. Regard and mood come from the Fellow's words against the sentiment anchors
 * (nearest anchor, weighted by how close it is — a weak match is ordinary talk, which is 0); without an
 * embedder they stay 0, which is the GDD default. Gift turns are 0: the engine already applied the
 * gift's Regard. memory/favor/reveal_secret stay null in v1 (GDD-GAP: the cheap path doesn't distill
 * memories yet).
 */
export async function cheapEffects(opts: {
  playerSaid: string;
  focus: TurnFocus;
  walkUp: boolean;
  wrapUp: boolean;
  embedder?: Embedder | null;
}): Promise<CheapDisposition> {
  const { playerSaid, focus, walkUp, wrapUp, embedder } = opts;
  let regard = 0;
  const scored = !walkUp && focus !== "gift" && embedder
    ? await embedder([playerSaid, ...SENTIMENT_ANCHORS.map((a) => a.text)]).then((vecs) =>
        SENTIMENT_ANCHORS.map((a, i) => ({ a, sim: cosine(vecs[0] ?? [], vecs[i + 1] ?? []) }))).catch(() => null)
    : null;
  if (scored) {
    const best = scored.reduce((m, s) => (s.sim > m.sim ? s : m), scored[0]);
    if (best && best.sim >= 0.3) {
      // Weight by closeness: a clear match takes the anchor's full delta, a borderline one half.
      regard = Math.max(-3, Math.min(3, Math.round(best.a.delta * Math.min(1, (best.sim - 0.25) * 2.5))));
    }
  }
  return {
    effects: {
      regard_delta: regard,
      mood: { valence: Math.max(-2, Math.min(2, regard)), stress: regard < 0 ? 1 : 0 },
      memory: null,
      favor: null,
      reveal_secret: null,
      topic: topicTag(focus),
    },
    wrapUp,
    topic: topicTag(focus),
  };
}
