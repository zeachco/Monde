/**
 * Cheap dialogue turns (GDD §9.10.2: "the model proposes; the engine disposes").
 *
 * generateTurn (dialogue.ts) routes here when the dialogue role resolves to a non-Claude model — a small
 * local or free model behind an OpenAI-compatible endpoint (MIND_PROVIDER=openai, e.g. Ollama's gemma4:e2b).
 * The small model is not asked to fill the structured DialogueTurnOutput: it streams the resident's lines as
 * plain text (the dialogue prompt ends "Reply as X.", no output format), and the engine disposes the rest:
 *  - choices: the authored candidate pool ranked by embedding similarity to what was just said (reactions.ts);
 *  - effects: regard and mood from the Fellow's words against sentiment anchors;
 *  - wrap_up: the forced last exchange (the note is what the route appends).
 * The assembled turn still passes validateTurn, so the same clamps and lexical guards apply as on the Claude
 * path. One call per turn, no hedge (a second parallel call on a local box doubles the load the hedge exists
 * to hide on a cold API connection), no salvage gymnastics: whole plain-text lines are kept if the stream
 * cuts.
 */
import { DialogueTurnOutput, LIMITS, fitText, splitLine } from "@monde/shared";
import { defaultRouter } from "./models.js";
import { tokensOf } from "./mindClient.js";
import { costUsd, type CallTokens } from "./pricing.js";
import { WALK_UP } from "./promptBuilder.js";
import { repairDialogueLine, validateTurn } from "./validate.js";
import {
  callOnce, DEFAULT_TIMINGS, dialogueParams, failure, streamSafe, textOf, WRAP_UP_NOTE,
  type Attempt, type CallRecord, type DialogueTimings, type GenerateOptions, type GenerateResult,
  type TurnFailureCode,
} from "./dialogue.js";
import { cheapEffects, lineExpression, pickChoices } from "./reactions.js";
import type { Embedder } from "./openaiClient.js";
import type { TurnFocus } from "./promptBuilder.js";

export interface CheapTurnOptions extends GenerateOptions {
  focus: TurnFocus;
  playerSaid: string;
  emotion?: string;
  embedder?: Embedder | null;
}

/** What the choice scorer reads: what the Fellow said and what the resident answered. */
const choiceQuery = (playerSaid: string, lines: string[]): string =>
  (playerSaid === WALK_UP ? "greeting" : playerSaid) + "\n" + lines.join(" ");

/**
 * Small models sometimes answer the plain instruction with a JSON object anyway: if the whole text parses
 * as one with a string `lines` array (or is a string array), use those lines instead of the raw text.
 */
function plainLines(text: string): string[] | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return null;
  try {
    const parsed: unknown = JSON.parse(trimmed);
    const raw = Array.isArray(parsed)
      ? parsed
      : parsed !== null && typeof parsed === "object" && Array.isArray((parsed as { lines?: unknown }).lines)
        ? (parsed as { lines: unknown[] }).lines
        : null;
    if (!raw) return null;
    const lines = raw.filter((line): line is string => typeof line === "string" && line.trim() !== "");
    return lines.length > 0 ? lines : null;
  } catch {
    return null; // not JSON after all: the raw text is the answer
  }
}

export async function cheapTurn(options: CheapTurnOptions): Promise<GenerateResult> {
  const router = options.router ?? defaultRouter;
  const selectedModel = router.resolve("dialogue").model;
  // No hedge: a second parallel call on a local box would double the load the hedge exists to hide on a
  // cold API connection. 1e9 ms (~11 days) keeps setTimeout in its 32-bit-safe range.
  const timings: DialogueTimings = { ...DEFAULT_TIMINGS, ...options.timings, hedgeMs: 1e9 };
  const now = options.now ?? (() => performance.now());
  const notes = options.notes ?? [];
  const calls: CallRecord[] = [];
  let visible = false;
  let finished = false;
  const thinkingTimer = setTimeout(() => {
    if (!visible && !finished) options.onThinking?.();
  }, timings.thinkingMs);
  const fail = (code: TurnFailureCode): GenerateResult => ({ output: null, failure: failure(code), calls, adjustments: [], salvaged: false });

  try {
    // Riddle-topic turns show only the authored hint wording (validate.ts step 5), same as the Claude path.
    if (options.validation.riddle) return fail("riddle_authored");

    // The builder's request with the structured output dropped: the endpoint answers in plain text.
    const params = (() => {
      const body = dialogueParams(options.request, notes, router, selectedModel);
      delete (body as { output_config?: unknown }).output_config;
      return body;
    })();

    const record = (tokens: CallTokens, stopReason: string, validation: string, result: Attempt): void => {
      calls.push({
        attempt: 1, model: params.model, tokens, costUsd: costUsd(tokens, params.model),
        latencyMs: result.latencyMs, firstTokenMs: result.firstTokenMs, stopReason, validation,
      });
    };

    // Plain-text streaming: a line is final when its newline arrives. Repair and gate it, then show it —
    // no JSON extraction, so there is nothing to hold back.
    const shown: string[] = [];
    let current = "";
    const emitLine = (text: string): void => {
      if (shown.length >= LIMITS.dialogue.maxLines) return;
      const repaired = repairDialogueLine(text, options.validation);
      if (!repaired.trim()) return;
      const final = repaired.length > LIMITS.dialogue.lineChars
        ? (splitLine(repaired, LIMITS.dialogue.lineChars)?.[0] ?? repaired)
        : fitText(repaired, LIMITS.dialogue.lineChars);
      if (!streamSafe(final, options.validation, shown)) return;
      if (final === shown[shown.length - 1]) return;
      shown.push(final);
      visible = true;
      options.onLineDelta?.({ index: shown.length - 1, text_so_far: final, attempt: 1 });
    };
    const onText = (delta: string): void => {
      const parts = (current + delta).split("\n");
      current = parts[parts.length - 1];
      for (const part of parts.slice(0, -1)) emitLine(part);
    };

    // The turn's structure, disposed by the engine once the lines are known.
    const dispose = async (lines: string[], tokens: CallTokens, result: Attempt, stopReason: string, validation: string, salvaged: boolean): Promise<GenerateResult> => {
      record(tokens, stopReason, validation, result);
      const choices = await pickChoices({ query: choiceQuery(options.playerSaid, lines), focus: options.focus, embedder: options.embedder ?? null });
      const wrapUp = notes.includes(WRAP_UP_NOTE);
      const disposition = await cheapEffects({
        playerSaid: options.playerSaid, focus: options.focus,
        walkUp: options.playerSaid === WALK_UP, wrapUp, embedder: options.embedder ?? null,
      });
      const output: DialogueTurnOutput = {
        lines: lines.map((text) => ({ text, expr: lineExpression(text, options.emotion) })),
        choices: choices.map((c, i) => ({ id: `c${i + 1}`, text: c.text, intent: c.intent })),
        effects: disposition.effects,
        wrap_up: wrapUp,
      };
      const raw = JSON.stringify(output);
      const validated = validateTurn(raw, options.validation);
      if (validated.ok) return { output: validated.output, rawOutput: raw, failure: null, calls, adjustments: validated.adjustments, salvaged };
      if (validated.soft) return {
        output: validated.soft.output, rawOutput: raw, failure: null, calls,
        adjustments: [...validated.soft.adjustments, `kept despite ${validated.category}: ${validated.detail}`], salvaged,
      };
      options.onHardFailure?.(`${validated.category}: ${validated.detail}`);
      return fail("invalid_output");
    };

    const result: Attempt = await callOnce(options.mind, params, timings, options.signal, onText, now);
    for (const extra of result.extra) {
      calls.push({ attempt: 1, model: params.model, costUsd: costUsd(extra.tokens, params.model), stopReason: "none", ...extra });
    }

    if (result.kind === "failed") {
      // It started and then stalled or dropped: the whole lines already streamed still make a turn.
      if (result.firstTokenMs !== null && result.code !== "aborted" && shown.length > 0) {
        return dispose(shown, result.tokens, result, "none", `salvaged:${result.code}`, true);
      }
      record(result.tokens, "none", result.code, result);
      return fail(result.code);
    }

    const { message } = result;
    const tokens = tokensOf(message.usage);
    const stopReason = message.stop_reason ?? "none";
    // Always check stop_reason before reading content (AGENTS.md).
    if (stopReason === "refusal") {
      record(tokens, stopReason, stopReason, result);
      return fail("refusal");
    }
    if (stopReason === "max_tokens") {
      // Plain text cut short: the whole lines are already shown; they make a turn if any.
      if (shown.length === 0) {
        record(tokens, stopReason, stopReason, result);
        return fail("max_tokens");
      }
      return dispose(shown, tokens, result, stopReason, "salvaged", true);
    }
    if (stopReason !== "end_turn" && stopReason !== "stop_sequence") {
      record(tokens, stopReason, `stop:${stopReason}`, result);
      return fail("invalid_output");
    }

    // The final text is authoritative; re-derive the lines with the same pipeline the stream used.
    const lines: string[] = [];
    for (const part of plainLines(textOf(message)) ?? textOf(message).split("\n")) {
      if (lines.length >= LIMITS.dialogue.maxLines) break;
      const repaired = repairDialogueLine(part, options.validation);
      if (!repaired.trim()) continue;
      const final = repaired.length > LIMITS.dialogue.lineChars
        ? (splitLine(repaired, LIMITS.dialogue.lineChars)?.[0] ?? repaired)
        : fitText(repaired, LIMITS.dialogue.lineChars);
      if (!streamSafe(final, options.validation, lines)) continue;
      if (final === lines[lines.length - 1]) continue;
      lines.push(final);
    }
    if (lines.length === 0) {
      record(tokens, stopReason, "invalid_output", result);
      return fail("invalid_output");
    }
    return dispose(lines, tokens, result, stopReason, "ok", false);
  } finally {
    finished = true;
    clearTimeout(thinkingTimer);
  }
}
