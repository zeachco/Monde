import { describe, expect, it } from 'vitest';
import Anthropic from '@anthropic-ai/sdk';
import { isClaudeModel, MODEL_ROLES, ModelRouter, modelParams } from '../models.js';

const openaiParams: Anthropic.MessageCreateParamsNonStreaming = {
  model: 'claude-haiku-5-5', max_tokens: 256,
  thinking: { type: 'disabled' },
  output_config: { effort: 'low', format: { type: 'json_schema' } as never },
  system: 'You are a resident.',
  messages: [{ role: 'user', content: 'hello' }],
};

describe('provider-aware model routing', () => {
  it('recognizes Claude ids and treats everything else as a plain endpoint model', () => {
    expect(isClaudeModel('claude-haiku-5-5')).toBe(true);
    expect(isClaudeModel('claude-fable-5-1')).toBe(true);
    expect(isClaudeModel('gemma4:e2b')).toBe(false);
    expect(isClaudeModel('qwen3.8')).toBe(false);
  });

  it('accepts any model name on the openai path and rejects it on the anthropic path', () => {
    expect(() => new ModelRouter({ MIND_PROVIDER: 'openai', MIND_MODEL: 'gemma4:e2b', MIND_MODEL_DIALOGUE: 'qwen3.8' })).not.toThrow();
    expect(() => new ModelRouter({ MIND_MODEL: 'gemma4:e2b' })).toThrow();
    expect(() => new ModelRouter({ MIND_PROVIDER: 'openai', MIND_MODEL: '' })).toThrow();
    expect(() => new ModelRouter({ MIND_PROVIDER: 'openai', MIND_MODEL_DIALOGUE: 'gemma4:e2b', MIND_MODEL_PLAN: '' })).toThrow();
  });

  it('resolves per-role env vars on the openai path and strips Claude-only params for non-Claude models', () => {
    const router = new ModelRouter({ MIND_PROVIDER: 'openai', MIND_MODEL: 'gemma4:e2b', MIND_MODEL_DIALOGUE: 'qwen3.8' });
    expect(router.resolve('dialogue')).toEqual({ model: 'qwen3.8', source: 'env' });
    expect(router.resolve('plan')).toEqual({ model: 'gemma4:e2b', source: 'env' });
    expect(router.strategy('qwen3.8')).toBe('disabled');

    const body = modelParams('dialogue', openaiParams, router, 'qwen3.8');
    expect(body.model).toBe('qwen3.8');
    expect(body).not.toHaveProperty('thinking');
    // The output format survives (the OpenAI client maps it to response_format) so the precheck keeps its
    // JSON shape; effort — a Claude thinking knob — does not.
    expect(body.output_config).toEqual({ format: { type: 'json_schema' } });
    expect(body.max_tokens).toBe(256);
    expect(body.messages).toEqual(openaiParams.messages);

    // An effort-only output_config (plan/reflect) is stripped entirely.
    const effort = modelParams('plan', { ...openaiParams, output_config: { effort: 'low' } as never }, router, 'qwen3.8');
    expect(effort).not.toHaveProperty('output_config');

    // The Claude path still applies its thinking/effort rules for a Claude model.
    const claude = modelParams('dialogue', openaiParams, router, 'claude-haiku-5-5');
    expect(claude.model).toBe('claude-haiku-5-5');
    expect(claude.thinking).toEqual({ type: 'disabled' });
    expect(claude.output_config?.effort).toBe('low');
  });

  it('keeps admin overrides provider-aware', async () => {
    const openai = new ModelRouter({ MIND_PROVIDER: 'openai', MIND_MODEL: 'gemma4:e2b' });
    (openai as unknown as { overrides: Record<string, string> }).overrides = { dialogue: 'llama3.2' };
    expect(openai.resolve('dialogue')).toEqual({ model: 'llama3.2', source: 'admin' });
    const anthropic = new ModelRouter({ MIND_MODEL: 'claude-haiku-5-5' });
    expect(() => (anthropic as unknown as { parseOverrides: (v: unknown) => unknown }).parseOverrides({ dialogue: 'llama3.2' })).toThrow();
    expect((anthropic as unknown as { parseOverrides: (v: unknown) => Record<string, string> }).parseOverrides({ dialogue: 'claude-sonnet-5-5' })).toEqual({ dialogue: 'claude-sonnet-5-5' });
  });

  it('lists every role in the status for both providers', () => {
    const openai = new ModelRouter({ MIND_PROVIDER: 'openai', MIND_MODEL: 'gemma4:e2b' });
    const status = openai.status();
    expect(Object.keys(status.roles).sort()).toEqual([...MODEL_ROLES].sort());
    for (const role of MODEL_ROLES) expect(status.roles[role].model).toBe('gemma4:e2b');
  });
});
