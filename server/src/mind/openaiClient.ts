/**
 * MindClient for any OpenAI-compatible endpoint (Ollama, llama.cpp's server, LM Studio), over raw fetch —
 * no SDK dependency (AGENTS.md: don't add dependencies). Claude-only parameters are dropped on the way in
 * (thinking, cache_control, output_config.effort); output_config.format becomes response_format json_object
 * where the endpoint supports it. The Batch API is not part of the OpenAI surface, so batch methods fail
 * with MindUnavailableError: a night job that needs it degrades to its fallback.
 *
 * Usage is best-effort: many local endpoints omit it, and a missing count reads as zero tokens (cost is
 * zero for non-Claude models anyway, pricing.ts). stop_reason maps finish_reason: stop→end_turn,
 * length→max_tokens, content_filter→refusal; an omitted finish_reason reads as end_turn.
 */
import type Anthropic from "@anthropic-ai/sdk";
import { MindStreamError, MindUnavailableError, type BatchOptions, type BatchRequest, type BatchResult, type BatchStatus, type MindClient, type StreamOptions } from "./mindClient.js";
import type { CallTokens } from "./pricing.js";

/** Ollama's OpenAI-compatible endpoint; MIND_BASE_URL overrides it. */
export const DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:11434/v1";

export interface OpenAIMindClientConfig {
  baseUrl?: string;
  apiKey?: string;
}

interface OAIUsage {
  input_tokens?: number;
  output_tokens?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
}

const usageOf = (u: OAIUsage | undefined): Anthropic.Usage => ({
  cache_creation: null,
  cache_creation_input_tokens: 0,
  cache_read_input_tokens: 0,
  inference_geo: null,
  input_tokens: u?.input_tokens ?? u?.prompt_tokens ?? 0,
  output_tokens: u?.output_tokens ?? u?.completion_tokens ?? 0,
  output_tokens_details: null,
  server_tool_use: null,
  service_tier: null,
});

const textBlock = (text: string): Anthropic.TextBlock => ({ type: "text", text, citations: null });

const messageOf = (model: string, text: string, stopReason: Anthropic.Message["stop_reason"], usage: OAIUsage | undefined): Anthropic.Message => ({
  id: `oai_${Date.now()}`, type: "message", role: "assistant", model,
  content: [textBlock(text)], stop_reason: stopReason, stop_sequence: null,
  container: null, diagnostics: null, stop_details: null,
  usage: usageOf(usage),
});

const mapFinish = (finish: string | null | undefined): Anthropic.Message["stop_reason"] =>
  finish === "length" ? "max_tokens" : finish === "content_filter" ? "refusal" : "end_turn";

/** Flatten Anthropic's block shapes to plain strings: text blocks kept, cache_control dropped. */
function flattenSystem(system: string | Anthropic.TextBlockParam[] | undefined): string | null {
  if (system == null) return null;
  if (typeof system === "string") return system.trim() === "" ? null : system;
  if (system.length === 0) return null;
  return system.map((b) => b.text).join("\n\n");
}

function flattenContent(content: Anthropic.MessageParam["content"]): string {
  if (typeof content === "string") return content;
  return content.map((b) => (b.type === "text" ? b.text : "")).join("\n\n");
}

/** The Anthropic request as an OpenAI chat-completions body. */
export function openaiBody(params: Anthropic.MessageCreateParamsNonStreaming, stream: boolean): Record<string, unknown> {
  const messages: { role: "system" | "user" | "assistant"; content: string }[] = [];
  const system = flattenSystem(params.system);
  if (system) messages.push({ role: "system", content: system });
  for (const m of params.messages) messages.push({ role: m.role, content: flattenContent(m.content) });
  const body: Record<string, unknown> = { model: params.model, messages, max_tokens: params.max_tokens, stream };
  if (params.output_config?.format) body.response_format = { type: "json_object" };
  return body;
}

/** `data:` payloads of an SSE stream, line by line, tolerant of a final chunk split across reads. */
async function* sseData(body: AsyncIterable<Uint8Array>): AsyncGenerator<string> {
  const decoder = new TextDecoder();
  let buf = "";
  for await (const chunk of body) {
    buf += decoder.decode(chunk, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line.startsWith("data:")) yield line.slice(5).trim();
    }
  }
  const tail = buf.trim();
  if (tail.startsWith("data:")) yield tail.slice(5).trim();
}

type OAIChunk = {
  choices?: { delta?: { content?: unknown }; message?: { content?: unknown }; finish_reason?: string | null }[];
  usage?: OAIUsage;
};

const NO_TOKENS: CallTokens = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

export class OpenAIMindClient implements MindClient {
  readonly enabled = true;
  private readonly base: string;
  private readonly headers: Record<string, string>;
  constructor(config: OpenAIMindClientConfig) {
    this.base = (config.baseUrl ?? DEFAULT_OPENAI_BASE_URL).replace(/\/+$/, "");
    this.headers = { "content-type": "application/json" };
    if (config.apiKey) this.headers.authorization = `Bearer ${config.apiKey}`;
  }

  private async request(path: string, body: unknown, signal?: AbortSignal): Promise<Response> {
    let res: Response;
    try {
      res = await fetch(this.base + path, { method: "POST", headers: this.headers, body: JSON.stringify(body), signal });
    } catch (error) {
      if (signal?.aborted) throw error; // The caller's deadline fired; keep the abort as the answer.
      throw new MindUnavailableError(`cannot reach the model endpoint at ${this.base}: ${error instanceof Error ? error.message : String(error)}`);
    }
    if (!res.ok) {
      const detail = (await res.text().catch(() => "")).slice(0, 300);
      if (res.status === 401 || res.status === 403) throw new MindUnavailableError(`the model endpoint refused the key (HTTP ${res.status})`);
      if (res.status === 404) throw new MindUnavailableError(`the model endpoint has no such model (HTTP 404): ${detail}`);
      throw new Error(`model endpoint HTTP ${res.status}: ${detail}`);
    }
    return res;
  }

  async streamMessage(params: Anthropic.MessageCreateParamsNonStreaming, options: StreamOptions): Promise<Anthropic.Message> {
    const res = await this.request("/chat/completions", openaiBody(params, true), options.signal);
    if (!res.body) throw new MindUnavailableError("the model endpoint returned an empty stream");
    let text = "";
    let finish: string | null | undefined;
    let usage: OAIUsage | undefined;
    try {
      for await (const data of sseData(res.body as unknown as AsyncIterable<Uint8Array>)) {
        if (data === "[DONE]") break;
        let chunk: OAIChunk;
        try {
          chunk = JSON.parse(data) as OAIChunk;
        } catch {
          continue; // Some endpoints send keep-alive lines; a payload that isn't JSON is not ours.
        }
        const choice = chunk.choices?.[0];
        const delta = choice?.delta?.content;
        if (typeof delta === "string" && delta) {
          text += delta;
          options.onText(delta);
        }
        if (choice?.finish_reason !== undefined) finish = choice.finish_reason;
        if (chunk.usage) usage = chunk.usage;
      }
    } catch (error) {
      throw new MindStreamError(error, text ? { ...NO_TOKENS, output: Math.ceil(text.length / 4) } : NO_TOKENS);
    }
    return messageOf(params.model, text, mapFinish(finish), usage);
  }

  async createMessage(params: Anthropic.MessageCreateParamsNonStreaming, options: { signal?: AbortSignal } = {}): Promise<Anthropic.Message> {
    const res = await this.request("/chat/completions", openaiBody(params, false), options.signal);
    const data = (await res.json()) as {
      choices?: { message?: { content?: unknown }; finish_reason?: string | null }[];
      usage?: OAIUsage;
    };
    const choice = data.choices?.[0];
    const text = typeof choice?.message?.content === "string" ? choice.message.content : "";
    return messageOf(params.model, text, mapFinish(choice?.finish_reason), data.usage);
  }

  listBatches(_options?: BatchOptions): AsyncIterable<BatchStatus & { created_at: string }> {
    throw new MindUnavailableError("the OpenAI provider has no batch API; the night pass degrades to its fallback");
  }
  async createBatch(_requests: BatchRequest[], _options?: BatchOptions): Promise<BatchStatus> {
    throw new MindUnavailableError("the OpenAI provider has no batch API");
  }
  async retrieveBatch(_id: string, _options?: BatchOptions): Promise<BatchStatus> {
    throw new MindUnavailableError("the OpenAI provider has no batch API");
  }
  batchResults(_id: string, _options?: BatchOptions): AsyncIterable<BatchResult> {
    throw new MindUnavailableError("the OpenAI provider has no batch API");
  }

  /** A Models list: free, and it leaves a warm connection behind for the first turn. */
  async warm(): Promise<void> {
    await fetch(this.base + "/models", { headers: this.headers, signal: AbortSignal.timeout(5000) }).then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    });
  }
}

/**
 * Batch embedding client for the cheap dialogue scorer (reactions.ts): one POST per 32 texts. Throws on
 * any failure; the scorer catches and runs rules-only.
 */
export type Embedder = (texts: string[]) => Promise<number[][]>;

export function createOpenAIEmbedder(baseUrl: string | undefined, model: string, apiKey?: string): Embedder {
  const base = (baseUrl ?? DEFAULT_OPENAI_BASE_URL).replace(/\/+$/, "");
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (apiKey) headers.authorization = `Bearer ${apiKey}`;
  return async (texts: string[]) => {
    const vectors: number[][] = [];
    for (let i = 0; i < texts.length; i += 32) {
      const res = await fetch(base + "/embeddings", {
        method: "POST", headers, body: JSON.stringify({ model, input: texts.slice(i, i + 32) }),
        signal: AbortSignal.timeout(10_000),
      });
      if (!res.ok) throw new Error(`embeddings endpoint HTTP ${res.status} (model ${model})`);
      const data = (await res.json()) as { data: { index?: number; embedding: number[] }[] };
      const sorted = [...data.data].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
      for (const item of sorted) vectors.push(item.embedding);
    }
    return vectors;
  };
}

export function createOpenAIMindClient(config: OpenAIMindClientConfig): MindClient {
  return new OpenAIMindClient(config);
}
