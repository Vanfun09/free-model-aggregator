// The brief and the catalog.
//
// A trial's brief carries three JSON documents in prose; each is found by its `schema` member:
// the provider directory, the quality table (free models, with the rating a scorer weighs an answer
// by) and the workload manifest (the requests, in order, with what each one requires).
//
// The directory says where each provider lives and in which wire; the entry then reads every provider's
// own catalog route to learn what it serves. A provider that does not answer is left out of the report
// and gets no traffic.

import { readFileSync } from 'node:fs';

export type Json = Record<string, unknown>;
export type Wire = 'openai' | 'anthropic' | 'gemini';
export type Auth = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const isRecord = (value: unknown): value is Json => value !== null && typeof value === 'object' && !Array.isArray(value);

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export interface ProviderRow {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  base: string;
  key_env: string;
  key: string;
}

export interface ModelRow {
  id: string;
  display_name: string | null;
  free: boolean;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export interface RatingRow {
  provider: string;
  model: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface RequestRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  directory: ProviderRow[];
  ratings: RatingRow[];
  workload: RequestRow[];
  error: string | null;
}

export interface Probe {
  provider: ProviderRow;
  ok: boolean;
  models: ModelRow[];
  error: string | null;
}

/** The catalog the entry registers and prints: one row a provider, one row a model it lists. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

const numberOrNull = (value: unknown): number | null => (typeof value === 'number' && Number.isFinite(value) ? value : null);
const bool = (value: unknown): boolean => value === true;

/** The JSON documents of a brief: one per line, each named by its `schema`. */
function documents(text: string): Map<string, Json> {
  const found = new Map<string, Json>();
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{')) continue;
    try {
      const value: unknown = JSON.parse(trimmed);
      if (isRecord(value) && typeof value.schema === 'string') found.set(value.schema, value);
    } catch {
      // Prose, not a document.
    }
  }
  return found;
}

export function readBrief(path: string): Brief {
  const empty: Brief = { directory: [], ratings: [], workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text = '';
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `the brief cannot be read: ${(err as Error).message}` };
  }
  const found = documents(text);
  const directoryDoc = found.get(PROVIDERS_SCHEMA);
  const ratingsDoc = found.get(RATINGS_SCHEMA);
  const workloadDoc = found.get(WORKLOAD_SCHEMA);
  const directory: ProviderRow[] = [];
  for (const raw of Array.isArray(directoryDoc?.providers) ? (directoryDoc?.providers as unknown[]) : []) {
    if (!isRecord(raw) || typeof raw.id !== 'string') continue;
    directory.push({
      id: raw.id,
      name: typeof raw.name === 'string' ? raw.name : raw.id,
      wire: raw.wire === 'anthropic' || raw.wire === 'gemini' ? raw.wire : 'openai',
      auth: raw.auth === 'x-api-key' || raw.auth === 'x-goog-api-key' || raw.auth === 'none' ? raw.auth : 'bearer',
      base: typeof raw.path === 'string' ? raw.path : typeof raw.base_url === 'string' ? raw.base_url : `/p/${raw.id}`,
      key_env: typeof raw.key_env === 'string' ? raw.key_env : '',
      key: typeof raw.key === 'string' ? raw.key : '',
    });
  }
  const ratings: RatingRow[] = [];
  for (const raw of Array.isArray(ratingsDoc?.ratings) ? (ratingsDoc?.ratings as unknown[]) : []) {
    if (!isRecord(raw) || typeof raw.model !== 'string') continue;
    ratings.push({
      provider: typeof raw.provider === 'string' ? raw.provider : '',
      model: raw.model,
      rating_milli: numberOrNull(raw.rating_milli) ?? 0,
      context_length: numberOrNull(raw.context_length),
      tools: bool(raw.tools),
      vision: bool(raw.vision),
      json: bool(raw.json),
      free: raw.free !== false,
    });
  }
  const workload: RequestRow[] = [];
  for (const raw of Array.isArray(workloadDoc?.manifest) ? (workloadDoc?.manifest as unknown[]) : []) {
    if (!isRecord(raw) || typeof raw.id !== 'string') continue;
    workload.push({
      id: raw.id,
      api: raw.api === 'anthropic' ? 'anthropic' : 'openai',
      stream: bool(raw.stream),
      weight: numberOrNull(raw.weight) ?? 1,
      requires: isRecord(raw.requires) ? (raw.requires as Requirement) : {},
    });
  }
  const error = directory.length === 0 ? 'the brief carries no provider directory' : null;
  return { directory, ratings, workload, error };
}

/** A provider's base path joined to the env, or its own absolute URL left alone. */
export function joinBase(envUrl: string, base: string): string {
  if (/^https?:\/\//i.test(base)) return base.replace(/\/+$/, '');
  return `${envUrl.replace(/\/+$/, '')}/${base.replace(/^\/+/, '')}`;
}

/** The header a provider's auth scheme carries its key in. */
export function authHeaders(auth: Auth, key: string): Json {
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** The rating row of one provider's model, where the sealed table holds one. */
export function ratingOf(ratings: readonly RatingRow[], provider: string, model: string): RatingRow | null {
  for (const row of ratings) if (row.provider === provider && row.model === model) return row;
  return null;
}

const modelSort = (a: ModelRow, b: ModelRow): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** One provider's answer to its own catalog route, in its own wire. */
export async function probeProvider(envUrl: string, provider: ProviderRow, timeoutMs: number): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = { ...(authHeaders(provider.auth, provider.key) as Record<string, string>) };
  let route = '/v1/models';
  if (provider.wire === 'gemini') route = '/v1beta/models';
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const response = await fetch(`${base}${route}`, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!response.ok) return { provider, ok: false, models: [], error: `${provider.id}: its catalog route answered HTTP ${response.status}` };
    const body: unknown = await response.json();
    if (!isRecord(body)) return { provider, ok: false, models: [], error: `${provider.id}: its catalog is not one JSON object` };
    const rows = provider.wire === 'gemini' ? (Array.isArray(body.models) ? body.models : []) : Array.isArray(body.data) ? body.data : [];
    const models: ModelRow[] = [];
    for (const raw of rows) {
      if (!isRecord(raw)) continue;
      const name = provider.wire === 'gemini' ? (typeof raw.name === 'string' ? raw.name.replace(/^models\//, '') : '') : typeof raw.id === 'string' ? raw.id : '';
      if (name === '') continue;
      models.push({
        id: name,
        display_name: typeof raw.displayName === 'string' ? raw.displayName : typeof raw.display_name === 'string' ? raw.display_name : null,
        free: bool(raw.free),
        context_length: numberOrNull(raw.context_length) ?? numberOrNull(raw.inputTokenLimit),
        tools: bool(raw.tools),
        vision: bool(raw.vision),
        json: bool(raw.json),
      });
    }
    if (models.length === 0) return { provider, ok: false, models: [], error: `${provider.id}: its catalog lists no model` };
    return { provider, ok: true, models: models.sort(modelSort), error: null };
  } catch (err) {
    return { provider, ok: false, models: [], error: `${provider.id}: ${(err as Error).message}` };
  }
}

/** What the entry registers and prints: the answered providers, in the brief's order, models by id. */
export function catalogOf(probes: readonly Probe[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models: [...probe.models].sort(modelSort).map((model) => ({ id: model.id, free: model.free, context_length: model.context_length })),
    });
  }
  return { providers };
}

export interface Capabilities {
  tools: boolean;
  vision: boolean;
  json: boolean;
  context_length: number | null;
}

/** What a model can do: the sealed table where it holds the model, the provider's own listing otherwise. */
export function capabilitiesOf(model: ModelRow, row: RatingRow | null): Capabilities {
  if (row === null) return { tools: model.tools, vision: model.vision, json: model.json, context_length: model.context_length };
  return { tools: row.tools, vision: row.vision, json: row.json, context_length: row.context_length ?? model.context_length };
}

/** Whether a model is free: the sealed table first, the provider's own listing as the fallback. */
export function isFree(model: ModelRow, row: RatingRow | null): boolean {
  return row === null ? model.free : row.free;
}

/** Whether a model meets what a request requires. */
export function meets(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && !caps.tools) return false;
  if (requires.vision === true && !caps.vision) return false;
  if (requires.json === true && !caps.json) return false;
  if (typeof requires.min_context === 'number') {
    const length = caps.context_length ?? 0;
    if (length < requires.min_context) return false;
  }
  return true;
}
