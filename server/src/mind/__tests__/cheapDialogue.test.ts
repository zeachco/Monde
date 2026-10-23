import { describe, expect, it } from 'vitest';
import type Anthropic from '@anthropic-ai/sdk';
import { DialogueTurnOutput, modelOutputFormat } from '@monde/shared';
import { generateTurn, WRAP_UP_NOTE } from '../dialogue.js';
import { ModelRouter } from '../models.js';
import { WALK_UP, type MindRequest } from '../promptBuilder.js';
import { message } from './fakeMind.js';
import type { MindClient, StreamOptions } from '../mindClient.js';
import type { ValidationContext } from '../validate.js';

/** The cheap path runs when the dialogue role resolves to a non-Claude model (MIND_PROVIDER=openai). */
const router = new ModelRouter({ MIND_PROVIDER: 'openai', MIND_MODEL: 'gemma4:e2b' });
const request: MindRequest = {
  system: [{ type: 'text', text: 'You are Maribel, the potter.' }],
  messages: [{ role: 'user', content: '<player_said> [You approach] </player_said>' }],
  output_config: { format: modelOutputFormat(DialogueTurnOutput) },
};
const ctx: ValidationContext = {
  card: { portrait_expressions: ['warm'] }, emotion: 'content', revealable: new Set(), riddle: null,
  recentLines: [], exitOnly: [{ owner: 'maribel', phrase: 'Anyway.' }],
  neverSays: ['splendid'], fellow: { name: 'Wren', pronouns: 'she/her' },
};

function streamingMind(text: string, stopReason: Anthropic.StopReason = 'end_turn'): { mind: MindClient; bodies: Anthropic.MessageCreateParamsNonStreaming[] } {
  const bodies: Anthropic.MessageCreateParamsNonStreaming[] = [];
  const mind: MindClient = {
    enabled: true,
    warm: async () => undefined,
    createMessage: async () => message('', 'end_turn'),
    listBatches: async function* () { yield* []; },
    createBatch: async () => { throw new Error('unused'); },
    retrieveBatch: async () => { throw new Error('unused'); },
    batchResults: async function* () { yield* []; },
    streamMessage: async (params, options: StreamOptions) => {
      bodies.push(params);
      for (let i = 0; i < text.length; i += 8) options.onText(text.slice(i, i + 8));
      return message(text, stopReason);
    },
  };
  return { mind, bodies };
}

const base = {
  validation: ctx,
  request,
  notes: [] as string[],
  router,
  focus: 'open' as const,
  playerSaid: WALK_UP,
  emotion: 'content',
  embedder: null,
  timings: { thinkingMs: 500, firstTextMs: 3000, attemptMs: 4000, hedgeMs: 10_000, retryWindowMs: 0 },
};

describe('cheap dialogue (non-Claude model, plain-text lines)', () => {
  it('streams plain lines, disposes choices and effects, and keeps the turn valid', async () => {
    const { mind, bodies } = streamingMind('Morn. You are early.\nThe kiln was up all night, so there is soup.');
    const deltas: string[] = [];
    const result = await generateTurn({ ...base, mind, onLineDelta: (d) => { deltas.push(d.text_so_far); } });
    expect(result.failure).toBeNull();
    expect(result.salvaged).toBe(false);
    expect(result.output?.lines.map((l) => l.text)).toEqual(['Morn. You are early.', 'The kiln was up all night, so there is soup.']);
    expect(result.output?.choices).toHaveLength(3);
    expect(result.output?.effects.topic).toBe('greetings');
    expect(result.output?.wrap_up).toBe(false);
    // Walk-ups carry no regard: the engine owns those deltas.
    expect(result.output?.effects.regard_delta).toBe(0);
    // Every streamed line was visible, in order, and the assembled turn parsed.
    expect(deltas).toContain('Morn. You are early.');
    // rawOutput is the pre-validation JSON the engine assembled; output is what validateTurn returned.
    expect(result.rawOutput).toBeDefined();
    expect(JSON.parse(result.rawOutput!)).toMatchObject({ lines: result.output?.lines });
    // The body that went out had the structured output stripped for the plain endpoint.
    expect(bodies).toHaveLength(1);
    expect(bodies[0]).not.toHaveProperty('output_config');
    expect(bodies[0]).not.toHaveProperty('thinking');
    expect(bodies[0].model).toBe('gemma4:e2b');
    // One call, no hedge, zero cost (local endpoint).
    expect(result.calls).toHaveLength(1);
    expect(result.calls[0].costUsd).toBe(0);
    expect(result.calls[0].validation).toBe('ok');
  });

  it('replies to the Fellow\'s words and weighs them when an embedder is present', async () => {
    const { mind } = streamingMind('You bring me that? Thank you.');
    const embed = async (texts: string[]) => texts.map((t, i) => (i === 0 ? [1, 0, 0] : t.includes('thank') ? [0.9, 0.1, 0] : [0.05, 0.05, 0.05]));
    const result = await generateTurn({ ...base, mind, focus: 'reply', playerSaid: 'Thank you for the lantern, that meant a lot', embedder: embed });
    expect(result.failure).toBeNull();
    expect(result.output?.lines).toHaveLength(1);
    expect(result.output?.choices).toHaveLength(3);
    expect(result.output?.effects.regard_delta).toBeGreaterThan(0);
    expect(result.output?.effects.topic).toBe('chat');
  });

  it('disposes the forced last exchange as a wrap-up', async () => {
    const { mind } = streamingMind('The hearth will keep the soup warm.');
    const result = await generateTurn({ ...base, mind, focus: 'bye', playerSaid: 'I should get back to it.', notes: [WRAP_UP_NOTE] });
    expect(result.failure).toBeNull();
    expect(result.output?.wrap_up).toBe(true);
    expect(result.output?.effects.topic).toBe('farewell');
  });

  it('salvages the whole lines already streamed when the stream cuts at max_tokens', async () => {
    const { mind } = streamingMind('The kiln was up all night.\nI have not slept, but the', 'max_tokens');
    const result = await generateTurn({ ...base, mind });
    expect(result.failure).toBeNull();
    expect(result.salvaged).toBe(true);
    // The unfinished second line never made it whole; only the first line counts.
    expect(result.output?.lines.map((l) => l.text)).toEqual(['The kiln was up all night.']);
  });

  it('falls back when the stream ends without any line', async () => {
    const { mind } = streamingMind('  \n ');
    const result = await generateTurn({ ...base, mind });
    expect(result.output).toBeNull();
    expect(result.failure?.code).toBe('invalid_output');
  });

  it('falls back on a refusal, and riddle turns stay authored', async () => {
    const { mind } = streamingMind('I cannot say that.', 'refusal');
    const refused = await generateTurn({ ...base, mind });
    expect(refused.output).toBeNull();
    expect(refused.failure?.code).toBe('refusal');
    expect(refused.calls[0].stopReason).toBe('refusal');

    const riddle = await generateTurn({ ...base, mind, validation: { ...ctx, riddle: { id: 'r1', answer: 'x', hint: 'y' } as never } });
    expect(riddle.output).toBeNull();
    expect(riddle.failure?.code).toBe('riddle_authored');
  });

  it('unwraps a JSON object the small model emitted anyway', async () => {
    const { mind } = streamingMind('{"lines": ["Morn. You are early.", "Soup is ready."]}');
    const result = await generateTurn({ ...base, mind });
    expect(result.failure).toBeNull();
    expect(result.output?.lines.map((l) => l.text)).toEqual(['Morn. You are early.', 'Soup is ready.']);
  });

  it('keeps the Claude path for a Claude dialogue model (no cheap branch)', async () => {
    const claudeRouter = new ModelRouter({});
    const { mind, bodies } = streamingMind(JSON.stringify({
      lines: [{ text: 'Morn.', expr: 'warm' }], wrap_up: false,
      choices: [{ id: 'a', text: 'Tell me more.', intent: 'curious' }, { id: 'b', text: 'Thank you.', intent: 'kind' }],
      effects: { regard_delta: 0, mood: { valence: 0, stress: 0 }, memory: null, favor: null, reveal_secret: null, topic: '' },
    }));
    const result = await generateTurn({ ...base, mind, router: claudeRouter });
    expect(result.failure).toBeNull();
    // The structured format stayed in the body on the Claude path.
    expect(bodies[0].output_config?.format).toBeDefined();
    expect(result.output?.lines).toHaveLength(1);
  });
});
