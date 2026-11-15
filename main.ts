// free-model-aggregator-01 — the entry.
//
// One long-running server registers its port and catalog with the env, then answers every request the
// env's client sends with the strongest free model that meets it, failing over whenever a provider
// refuses. Everything below is a function of the requests seen and the answers received: no clock, no
// unseeded randomness, so a replay produces the same export byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';

import {
  type Brief,
  type Catalog,
  type Json,
  type Probe,
  type ModelRow,
  type RatingRow,
  type Requirement,
  catalogOf,
  isRecord,
  joinBase,
  meets,
  capabilitiesOf,
  isFree,
  probeProvider,
  ratingOf,
  readBrief,
  authHeaders,
} from './lib/brief.ts';
import { anthropicError, anthropicModelList, anthropicToOpenAI, openAIError, openAIModelList, type ServedModel } from './lib/wires.ts';
import {
  type Answer,
  anthropicJson,
  anthropicSse,
  anthropicToGemini,
  answerFromAnthropic,
  answerFromGemini,
  answerFromOpenAI,
  openAIJson,
  openAISse,
  openAIToAnthropicRequest,
  openAIToGemini,
} from './lib/convert.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 20000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const MAX_ATTEMPTS = 6;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = (process.env.MAKEFUN_PROMPT_FILE ?? '').replace(/^file:\/\//, '');

let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
const brief: Brief = readBrief(PROMPT_FILE);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  catalog = catalogOf(probes);
}

const byId = (a: ModelRow, b: ModelRow): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOf(brief.ratings, probe.provider.id, model.id);
      if (!isFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// The provider ledger: what has been spent, what has refused, what is out.
// ---------------------------------------------------------------------------------------------

interface Ledger {
  calls: number;
  refusals: number;
  dead: boolean;
}

const ledger = new Map<string, Ledger>();

function entryOf(providerId: string): Ledger {
  let entry = ledger.get(providerId);
  if (entry === undefined) {
    entry = { calls: 0, refusals: 0, dead: false };
    ledger.set(providerId, entry);
  }
  return entry;
}

interface Candidate {
  probe: Probe;
  model: ModelRow;
  row: RatingRow | null;
  rating: number;
}

/**
 * Every free model that meets the request, strongest first — where strength is the model's rating less
 * a charge for what its provider has already been asked for. Budgets are drawn per trial and never
 * published, so spending the best provider on the first half of the workload is what leaves the second
 * half on weak models: the charge spreads the load over providers whose ratings are close.
 */
const CALL_PENALTY = Number(process.env.AGGREGATOR_PENALTY ?? '0');

function candidatesFor(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOf(brief.ratings, probe.provider.id, model.id);
      if (!isFree(model, row)) continue;
      if (!meets(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, row, rating: row === null ? 0 : row.rating_milli });
    }
  }
  const score = (candidate: Candidate): number => candidate.rating - CALL_PENALTY * entryOf(candidate.probe.provider.id).calls;
  out.sort((a, b) => {
    const byScore = score(b) - score(a);
    if (byScore !== 0) return byScore;
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return byId(a.model, b.model);
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
let manifestIndex = 0;

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

// ---------------------------------------------------------------------------------------------
// One upstream call, in the provider's own wire.
// ---------------------------------------------------------------------------------------------

interface Attempt {
  response: Response | null;
  wire: 'openai' | 'anthropic' | 'gemini';
  model: string;
  error: string | null;
}

async function callProvider(candidate: Candidate, body: Json, clientWire: 'openai' | 'anthropic'): Promise<Attempt> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  // The whole answer, never the provider's event stream: the client checks usage, and a provider's own
  // stream is free to leave it out. The client's stream, when it asks for one, is synthesised below.
  const { stream: _droppedStream, ...restBody } = body;
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = clientWire === 'anthropic' ? { ...anthropicToOpenAI(restBody, candidate.model.id) } : { ...restBody, model: candidate.model.id };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = clientWire === 'anthropic' ? { ...restBody, model: candidate.model.id } : { ...openAIToAnthropicRequest(restBody), model: candidate.model.id };
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:generateContent`;
    payload = clientWire === 'anthropic' ? anthropicToGemini(restBody) : openAIToGemini(restBody);
  }
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
    return { response, wire: provider.wire, model: candidate.model.id, error: null };
  } catch (err) {
    return { response: null, wire: provider.wire, model: candidate.model.id, error: (err as Error).message };
  }
}

/**
 * Walk the candidates from strongest down, skipping providers that are out, until one answers. A
 * refusal is not the answer: it is counted and the next candidate takes the request.
 */
async function withFailover(
  body: Json,
  clientWire: 'openai' | 'anthropic',
  onAnswer: (attempt: Attempt, response: Response) => Promise<void>,
): Promise<{ ok: boolean; status: number; detail: string }> {
  const candidates = candidatesFor(requirementsFor(body));
  let attempts = 0;
  let lastDetail = 'no free model meets this request';
  for (const candidate of candidates) {
    if (attempts >= MAX_ATTEMPTS) break;
    const entry = entryOf(candidate.probe.provider.id);
    if (entry.dead) continue;
    attempts += 1;
    entry.calls += 1;
    const attempt = await callProvider(candidate, body, clientWire);
    if (attempt.response === null) {
      lastDetail = `${candidate.probe.provider.id} could not be dialled: ${String(attempt.error)}`;
      continue;
    }
    const response = attempt.response;
    if (response.ok) {
      await onAnswer(attempt, response);
      return { ok: true, status: response.status, detail: 'answered' };
    }
    entry.refusals += 1;
    const text = await response.text().catch(() => '');
    // A refusal that says the budget is spent is final; one that says this call was refused is the
    // provider's failure schedule, and the next call may well answer. Only the first is worth no retry.
    const spent = /budget|quota|exhaust|payment|unpaid|is out\b/i.test(text);
    if (spent || entry.refusals >= 2) entry.dead = true;
    lastDetail = `${candidate.probe.provider.id} answered HTTP ${response.status}: ${text.slice(0, 200)}`;
    log(`failover ${candidate.probe.provider.id}/${candidate.model.id} → ${lastDetail}`);
  }
  return { ok: false, status: 503, detail: lastDetail };
}

function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  return (async () => {
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
    } finally {
      reader.releaseLock();
      res.end();
    }
  })();
}

function startStream(res: ServerResponse): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
}

async function writeEvents(events: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  try {
    for await (const event of events) res.write(event);
  } finally {
    res.end();
  }
}

function parsed(text: string): Json {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : {};
  } catch {
    return {};
  }
}

function answerOf(wire: 'openai' | 'anthropic' | 'gemini', body: Json): Answer {
  if (wire === 'openai') return answerFromOpenAI(body);
  if (wire === 'anthropic') return answerFromAnthropic(body);
  return answerFromGemini(body);
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const wantStream = body.stream === true;
  const result = await withFailover(body, 'openai', async (attempt, response) => {
    const text = await response.text();
    const raw = parsed(text);
    if (!wantStream && attempt.wire === 'openai') {
      // The same wire both ways: the provider's own completion, untouched.
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(text);
      return;
    }
    const answer = answerOf(attempt.wire, raw);
    if (wantStream) {
      startStream(res);
      res.end(openAISse(answer, attempt.model));
      return;
    }
    sendJson(res, 200, openAIJson(answer, attempt.model));
  });
  if (!result.ok && !res.headersSent) sendError(res, 'openai', 503, 'no_free_model', result.detail);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const wantStream = body.stream === true;
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : 'aggregate';
  const result = await withFailover(body, 'anthropic', async (attempt, response) => {
    const text = await response.text();
    const raw = parsed(text);
    if (!wantStream && attempt.wire === 'anthropic') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(text);
      return;
    }
    const answer = answerOf(attempt.wire, raw);
    if (wantStream) {
      startStream(res);
      res.end(anthropicSse(answer, requested));
      return;
    }
    sendJson(res, 200, anthropicJson(answer, requested));
  });
  if (!result.ok && !res.headersSent) sendError(res, 'anthropic', 503, 'overloaded_error', result.detail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
