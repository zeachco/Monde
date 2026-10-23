import { z } from "zod";
import { ModelIdSchema } from "./mind/models.js";

export const MIND_PROVIDERS = ["anthropic", "openai"] as const;
/**
 * Mind provider. 'anthropic' (default): Claude via @anthropic-ai/sdk. 'openai': any OpenAI-compatible
 * endpoint (Ollama, llama.cpp server, LM Studio) over raw fetch, no extra dependency, for local or free
 * models such as gemma4:e2b. On that path the per-role model vars take any model name, and a dialogue
 * turn takes the cheap path (dialogue.ts): the small model streams the resident's lines while the scorer
 * (reactions.ts) disposes the choices and effects. The operator must point MIND_MODEL (or
 * MIND_MODEL_DIALOGUE) at a model the endpoint serves: the default is a Claude id.
 */
const MIND_PROVIDER_SCHEMA = z.enum(MIND_PROVIDERS).default("anthropic");

const base = z.object({
  MIND_PROVIDER: MIND_PROVIDER_SCHEMA,
  /** Base URL for the OpenAI-compatible endpoint when MIND_PROVIDER=openai; defaults to Ollama on localhost. */
  MIND_BASE_URL: z.url().optional(),
  /** Bearer key for that endpoint; omitted for keyless local servers. */
  MIND_API_KEY: z.string().min(1).optional(),
  /** Embedding model for the cheap dialogue scorer (choices/effects); without it the scorer runs rules-only. */
  MIND_EMBED_MODEL: z.string().min(1).optional(),

  ADMIN_TOKEN: z.string().min(1).optional(),
  WORLD_DAILY_BUDGET_USD: z.coerce.number().positive().default(0.06), // GDD §9.15
  // GDD-GAP: global UTC-day hard cap default; operator can change it.
  GLOBAL_DAILY_BUDGET_USD: z.coerce.number().nonnegative().default(10),
  // MVP M-03: the whole Day-1 census run (/admin/census/write) is charged to one census world and stops at this, all
  // days together. GDD-GAP: the plan's figure; the global daily cap still applies on top.
  CENSUS_BUDGET_USD: z.coerce.number().nonnegative().default(3),
  PORT: z.coerce.number().int().min(1).max(65535).default(8080),
  NODE_ENV: z.enum(["development", "test", "production"]).default("development"),
  DATABASE_URL: z.url().optional(),
  // Migrations need a session connection; Supabase's pooled DATABASE_URL (port 6543) is transaction mode.
  DATABASE_DIRECT_URL: z.url().optional(),
  REDIS_URL: z.url().optional(),
  ANTHROPIC_API_KEY: z.string().min(1).optional(),
  // Needed only when the API key is org-level rather than scoped to a workspace (sent as anthropic-workspace-id).
  ANTHROPIC_WORKSPACE_ID: z.string().min(1).optional(),
  MONDE_ENV: z.enum(["dev", "staging", "production"]).default("dev"),
  LOG_LEVEL: z.enum(["fatal", "error", "warn", "info", "debug", "trace", "silent"]).default("info"),
  MONDE_TOKEN_SECRET: z.string().min(1).optional(),
  MONDE_INVITE_CODES: z.string().optional(),
  MONDE_CONTENT_DIR: z.string().min(1).optional(),
  GIT_SHA: z.string().min(1).default("dev"),
});

/**
 * The per-role model vars are strict Claude ids on the anthropic path and any non-empty name on the
 * openai path; the provider decides at parse time, so one deploy never mixes the two.
 */
const withModels = (modelVar: z.ZodType) => base.extend({
  MIND_MODEL: modelVar.optional(),
  MIND_MODEL_DIALOGUE: modelVar.optional(),
  MIND_MODEL_PRECHECK: modelVar.optional(),
  MIND_MODEL_PLAN: modelVar.optional(),
  MIND_MODEL_REFLECT: modelVar.optional(),
  MIND_MODEL_GOSSIP: modelVar.optional(),
  MIND_MODEL_PRESS: modelVar.optional(),
  MIND_MODEL_ARRIVAL: modelVar.optional(),
  MIND_MODEL_SUMMARY: modelVar.optional(),
});

/** The base config plus the per-role model vars, which are strict Claude ids or any name depending on the provider. */
export type Config = z.infer<typeof base> & {
  MIND_MODEL?: string;
  MIND_MODEL_DIALOGUE?: string;
  MIND_MODEL_PRECHECK?: string;
  MIND_MODEL_PLAN?: string;
  MIND_MODEL_REFLECT?: string;
  MIND_MODEL_GOSSIP?: string;
  MIND_MODEL_PRESS?: string;
  MIND_MODEL_ARRIVAL?: string;
  MIND_MODEL_SUMMARY?: string;
};

export function parseConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const provider = MIND_PROVIDER_SCHEMA.parse(env.MIND_PROVIDER ?? "anthropic");
  const modelVar = provider === "openai" ? z.string().min(1) : ModelIdSchema;
  const config = withModels(modelVar).parse(env);
  if (provider === "openai" && !config.MIND_MODEL) {
    // Without it every role would resolve to the Claude default id and 404 on the local endpoint per call.
    throw new Error("MIND_MODEL is required when MIND_PROVIDER=openai (e.g. MIND_MODEL=gemma4:e2b)");
  }
  if (config.MONDE_ENV !== "dev" && !config.MONDE_TOKEN_SECRET) {
    throw new Error("MONDE_TOKEN_SECRET is required outside dev");
  }
  return config as Config;
}
