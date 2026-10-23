import { afterEach, describe, expect, it, vi } from 'vitest';
import type Anthropic from '@anthropic-ai/sdk';
import {
  createOpenAIEmbedder, createOpenAIMindClient, DEFAULT_OPENAI_BASE_URL, openaiBody,
} from '../openaiClient.js';
import { MindUnavailableError } from '../mindClient.js';

const params = (over: Partial<Anthropic.MessageCreateParamsNonStreaming> = {}): Anthropic.MessageCreateParamsNonStreaming => ({
  model: 'gemma4:e2b',
  max_tokens: 256,
  system: [{ type: 'text', text: 'You are a resident.', cache_control: { type: 'ephemeral' } }],
  messages: [
    { role: 'user', content: [{ type: 'text', text: 'Morning.' }, { type: 'text', text: 'How is the kiln?' }] },
    { role: 'assistant', content: 'Morn.' },
    { role: 'user', content: '[You hand over a gift]' },
  ],
  ...over,
});

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
const sse = (payloads: string[]) => new Response(new ReadableStream<Uint8Array>({
  start(controller) {
    for (const p of payloads) controller.enqueue(new TextEncoder().encode(p));
    controller.close();
  },
}), { headers: { 'content-type': 'text/event-stream' } });

afterEach(() => vi.unstubAllGlobals());

describe('openaiBody', () => {
  it('flattens Anthropic blocks to plain messages and drops cache_control', () => {
    const body = openaiBody(params(), false) as { model: string; max_tokens: number; stream: boolean; messages: { role: string; content: string }[] };
    expect(body.model).toBe('gemma4:e2b');
    expect(body.max_tokens).toBe(256);
    expect(body.stream).toBe(false);
    expect(body.messages).toEqual([
      { role: 'system', content: 'You are a resident.' },
      { role: 'user', content: 'Morning.\n\nHow is the kiln?' },
      { role: 'assistant', content: 'Morn.' },
      { role: 'user', content: '[You hand over a gift]' },
    ]);
    expect(JSON.stringify(body)).not.toContain('cache_control');
  });

  it('maps output_config.format to response_format json_object, and nothing else of it', () => {
    const body = openaiBody(params({ output_config: { effort: 'low', format: { type: 'json_schema' } as never } }) as Anthropic.MessageCreateParamsNonStreaming, true) as { response_format?: unknown; stream: boolean };
    expect(body.response_format).toEqual({ type: 'json_object' });
    expect(body.stream).toBe(true);
    expect(JSON.stringify(body)).not.toContain('effort');
    expect(openaiBody(params(), true)).not.toHaveProperty('response_format');
  });
});

describe('OpenAIMindClient', () => {
  it('maps usage and finish_reason on a plain call', async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      expect(String(url)).toBe(`${DEFAULT_OPENAI_BASE_URL}/chat/completions`);
      const body = JSON.parse(String(init?.body)) as { stream: boolean };
      expect(body.stream).toBe(false);
      return json({
        choices: [{ message: { content: 'Morn.' }, finish_reason: 'length' }],
        usage: { prompt_tokens: 11, completion_tokens: 2 },
      });
    });
    vi.stubGlobal('fetch', fetchMock);
    const mind = createOpenAIMindClient({});
    const message = await mind.createMessage(params());
    expect(message.stop_reason).toBe('max_tokens');
    expect(message.content[0]).toMatchObject({ type: 'text', text: 'Morn.' });
    expect(message.usage.input_tokens).toBe(11);
    expect(message.usage.output_tokens).toBe(2);

    fetchMock.mockImplementation(async () => json({ choices: [{ message: { content: 'x' }, finish_reason: 'content_filter' }] }));
    expect((await mind.createMessage(params())).stop_reason).toBe('refusal');
    fetchMock.mockImplementation(async () => json({ choices: [{ message: { content: 'x' } }], usage: { input_tokens: 3, output_tokens: 4 } }));
    const ended = await mind.createMessage(params());
    expect(ended.stop_reason).toBe('end_turn');
    expect(ended.usage.input_tokens).toBe(3);
    expect(ended.usage.output_tokens).toBe(4);
  });

  it('streams SSE deltas, ignores keep-alive lines, and maps the final finish_reason', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => sse([
      ': keep-alive\n',
      'data: {"choices":[{"delta":{"content":"Morn."}}]}\n',
      'data: {"choices":[{"delta":{"content":" You are early."}}]}\n\ndata: not-json-keepalive\n',
      'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"input_tokens":7,"output_tokens":2}}\n',
      'data: [DONE]\n',
    ])));
    const mind = createOpenAIMindClient({});
    const deltas: string[] = [];
    const message = await mind.streamMessage(params(), { signal: new AbortController().signal, onText: (d) => { deltas.push(d); } });
    expect(deltas).toEqual(['Morn.', ' You are early.']);
    expect(message.content[0]).toMatchObject({ type: 'text', text: 'Morn. You are early.' });
    expect(message.stop_reason).toBe('end_turn');
    expect(message.usage.input_tokens).toBe(7);
    expect(message.usage.output_tokens).toBe(2);
  });

  it('survives a data payload split across reads', async () => {
    const payload = 'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n';
    vi.stubGlobal('fetch', vi.fn(async () => sse([payload.slice(0, 9), payload.slice(9)])));
    const mind = createOpenAIMindClient({});
    const message = await mind.streamMessage(params(), { signal: new AbortController().signal, onText: () => undefined });
    expect(message.content[0]).toMatchObject({ text: 'hi' });
  });

  it('maps transport failures to MindUnavailableError and keeps the caller abort', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('ECONNREFUSED'); }));
    const mind = createOpenAIMindClient({});
    await expect(mind.createMessage(params())).rejects.toBeInstanceOf(MindUnavailableError);

    vi.stubGlobal('fetch', vi.fn(async () => json({ error: 'no such model' }, 404)));
    await expect(mind.createMessage(params())).rejects.toThrow(/no such model/);

    vi.stubGlobal('fetch', vi.fn(async () => json({ error: 'bad key' }, 401)));
    await expect(mind.createMessage(params())).rejects.toBeInstanceOf(MindUnavailableError);

    vi.stubGlobal('fetch', vi.fn(async () => json({ error: 'boom' }, 500)));
    await expect(mind.createMessage(params())).rejects.toThrow(/HTTP 500/);

    const controller = new AbortController();
    vi.stubGlobal('fetch', vi.fn(async (_url: string, init?: RequestInit) => new Promise<Response>((_, reject) => {
      if (init?.signal?.aborted) { reject(new Error('aborted')); return; }
      init?.signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
    })));
    controller.abort();
    await expect(mind.createMessage(params(), { signal: controller.signal })).rejects.toThrow('aborted');
  });

  it('sends the API key when configured and has no batch API', async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      expect(new Headers(init?.headers).get('authorization')).toBe('Bearer test-key');
      return json({ choices: [{ message: { content: 'x' } }] });
    });
    vi.stubGlobal('fetch', fetchMock);
    const mind = createOpenAIMindClient({ apiKey: 'test-key' });
    await expect(mind.createMessage(params())).resolves.toMatchObject({ stop_reason: 'end_turn' });

    expect(() => mind.listBatches()).toThrow(MindUnavailableError);
    await expect(mind.createBatch([])).rejects.toBeInstanceOf(MindUnavailableError);
    expect(() => mind.batchResults('b_1')).toThrow(MindUnavailableError);
  });

  it('warms through the models endpoint', async () => {
    const fetchMock = vi.fn(async (url: string) => { expect(String(url)).toBe(`${DEFAULT_OPENAI_BASE_URL}/models`); return json({ data: [] }); });
    vi.stubGlobal('fetch', fetchMock);
    await expect(createOpenAIMindClient({}).warm?.()).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe('createOpenAIEmbedder', () => {
  it('batches 32 at a time and sorts by index', async () => {
    const calls: { model: string; input: string[] }[] = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      expect(String(url)).toBe('http://127.0.0.1:11434/v1/embeddings');
      const body = JSON.parse(String(init?.body)) as { model: string; input: string[] };
      calls.push(body);
      // Return out of order on purpose; the embedder must sort by index.
      return json({ data: body.input.map((_, i) => ({ index: (i + 1) % body.input.length, embedding: [i + 1, 0] })) });
    }));
    const embed = createOpenAIEmbedder('http://127.0.0.1:11434/v1', 'nomic-embed-text');
    const texts = Array.from({ length: 33 }, (_, i) => `text ${i}`);
    const vectors = await embed(texts);
    expect(calls.map((c) => c.input.length)).toEqual([32, 1]);
    expect(calls[0].model).toBe('nomic-embed-text');
    expect(vectors).toHaveLength(33);
    // The mock returns out of order; after sorting, position i holds the embedding the mock tagged index i with.
    expect(vectors[0]).toEqual([32, 0]);
    for (let i = 1; i < 32; i++) expect(vectors[i]).toEqual([i, 0]);
    expect(vectors[32]).toEqual([1, 0]);
  });

  it('throws on a bad response so the scorer falls back to rules', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json({ error: 'no model' }, 404)));
    const embed = createOpenAIEmbedder(undefined, 'nomic-embed-text');
    await expect(embed(['a'])).rejects.toThrow(/HTTP 404/);
  });
});
