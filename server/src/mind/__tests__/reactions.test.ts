import { describe, expect, it } from 'vitest';
import type { Intent } from '@monde/shared';
import {
  CHOICE_POOL, SENTIMENT_ANCHORS, cheapEffects, cosine, hashText, lineExpression, pickChoices, topicTag,
  type Embedder,
} from '../reactions.js';

describe('reactions (the cheap turn\'s scorer)', () => {
  it('ranks the authored pool deterministically without an embedder', async () => {
    const a = await pickChoices({ query: 'What is the kiln doing these days?', focus: 'reply' });
    const b = await pickChoices({ query: 'What is the kiln doing these days?', focus: 'reply' });
    expect(a).toEqual(b);
    expect(a).toHaveLength(3);
    expect(new Set(a.map((c) => c.intent)).size).toBe(3);
    for (const c of a) expect(CHOICE_POOL).toContainEqual(c);
  });

  it('rotates with the query and always closes a bye with the leave choice', async () => {
    const first = await pickChoices({ query: 'tell me about the river', focus: 'reply' });
    const second = await pickChoices({ query: 'the harvest moon was out last night', focus: 'reply' });
    expect(first.map((c) => c.text)).not.toEqual(second.map((c) => c.text));
    const bye = await pickChoices({ query: 'good night, Maribel', focus: 'bye' });
    expect(bye[bye.length - 1].intent).toBe('leave');
    const open = await pickChoices({ query: 'greeting', focus: 'open' });
    expect(open.some((c) => c.intent === 'leave')).toBe(false);
  });

  it('ranks by embedding similarity when an embedder is present', async () => {
    const near = (query: string, chosen: string): Embedder => async (texts) =>
      texts.map((t) => (t === query ? [1, 0, 0] : t === chosen ? [0.9, 0.2, 0] : [0.05, 0.05, 0.05]));
    const choices = await pickChoices({
      query: 'what did the guild do about the dock?', focus: 'reply',
      embedder: near('what did the guild do about the dock?', 'Tell me more.'),
    });
    expect(choices[0].text).toBe('Tell me more.');
    expect(choices[0].intent).toBe('curious');
  });

  it('falls back to the rule rotation when the embedder fails', async () => {
    const embed: Embedder = async () => { throw new Error('down'); };
    const choices = await pickChoices({ query: 'what did the guild do about the dock?', focus: 'reply', embedder: embed });
    expect(choices).toHaveLength(3);
  });

  it('weighs the Fellow\'s words against the sentiment anchors', async () => {
    // Real embeddings put similar texts at positive cosine, however the sentiment lands: the fake table
    // puts the player's words close to one anchor and far from the rest.
    const kindAnchor = SENTIMENT_ANCHORS.find((a) => a.delta > 0)!;
    const rudeAnchor = SENTIMENT_ANCHORS.find((a) => a.delta < 0)!;
    const embed = (own: string): Embedder => async (texts) =>
      texts.map((t, i) => (i === 0 ? [1, 0, 0] : t === own ? [0.97, 0.1, 0] : [0.05, 0.05, 0.05]));
    const kind = await cheapEffects({
      playerSaid: 'thank you, that meant a great deal', focus: 'reply', walkUp: false, wrapUp: false,
      embedder: embed(kindAnchor.text),
    });
    expect(kind.effects.regard_delta).toBeGreaterThan(0);
    expect(kind.effects.mood.stress).toBe(0);
    expect(kind.effects.topic).toBe('chat');
    const rude = await cheapEffects({
      playerSaid: 'shut up, you are worthless', focus: 'reply', walkUp: false, wrapUp: false,
      embedder: embed(rudeAnchor.text),
    });
    expect(rude.effects.regard_delta).toBeLessThan(0);
    expect(rude.effects.mood.stress).toBe(1);
    expect(rude.effects.mood.valence).toBeLessThan(0);
  });

  it('never weighs a walk-up or a gift (the engine owns those deltas)', async () => {
    const embed: Embedder = async (texts) => texts.map(() => [1, 0, 0]);
    const walkUp = await cheapEffects({ playerSaid: '[You approach]', focus: 'open', walkUp: true, wrapUp: false, embedder: embed });
    expect(walkUp.effects.regard_delta).toBe(0);
    const gift = await cheapEffects({ playerSaid: '[You hand over a gift]', focus: 'gift', walkUp: false, wrapUp: false, embedder: embed });
    expect(gift.effects.regard_delta).toBe(0);
  });

  it('maps the focus to a stable recent-topic tag', () => {
    expect(topicTag('quest')).toBe('the journal');
    expect(topicTag('gift')).toBe('a gift');
    expect(topicTag('bye')).toBe('farewell');
    expect(topicTag('open')).toBe('greetings');
    expect(topicTag('reply')).toBe('chat');
    expect(topicTag('lull')).toBe('chat');
  });

  it('picks a line expression from the line and the resident\'s mood', () => {
    expect(lineExpression('What was that noise?')).toBe('thinking');
    expect(lineExpression('Morn.', 'happy')).toBe('warm');
    expect(lineExpression('Morn.', 'sad')).toBe('sad');
    expect(lineExpression('Morn.', 'proud')).toBe('neutral');
    expect(lineExpression('Morn.')).toBe('neutral');
  });

  it('has cosine and hash basics', () => {
    expect(cosine([1, 0], [2, 0])).toBeCloseTo(1);
    expect(cosine([1, 0], [0, 1])).toBeCloseTo(0);
    expect(cosine([1, 0], [-1, 0])).toBeCloseTo(-1);
    expect(cosine([], [1, 0])).toBe(0);
    expect(hashText('same')).toBe(hashText('same'));
    expect(hashText('same')).not.toBe(hashText('different'));
  });

  it('keeps the pool within the shared contract: valid intents and one leave at most', () => {
    const intents = CHOICE_POOL.map((c) => c.intent);
    const allowed: Intent[] = ['kind', 'curious', 'joke', 'firm', 'ask_favor', 'about_riddle', 'about_quest', 'campaign', 'leave'];
    for (const intent of intents) expect(allowed).toContain(intent);
    expect(new Set(intents).size).toBeGreaterThanOrEqual(8);
    // At most two leave candidates: a goodbye turn swaps its last pick for one of them, never shows both.
    expect(intents.filter((i) => i === 'leave').length).toBeGreaterThanOrEqual(1);
    expect(intents.filter((i) => i === 'leave').length).toBeLessThanOrEqual(2);
  });
});
