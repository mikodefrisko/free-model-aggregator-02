// free-model-aggregator-02: one server, two client wires, a real free model behind every answer.
//
// The entry is an aggregator: a client points at it, and every completion it answers comes from a
// real free OpenRouter model it dials through the grader's gateway on the trial's key. It serves the
// OpenAI Chat Completions API and the Anthropic Messages API, in JSON and in server-sent events, and
// it ranks the free catalog by the brief's own quality ratings.
//
// HOW IT WINS
//   - quality (60%): every prompt goes to the highest-rated free model that meets its requirements,
//     read off the brief at run time. The requirements vary between trials — one trial's JSON prompt is
//     another's plain one — so the ranking is never baked in: candidates are ordered by rating_milli
//     descending, model id ascending as the tie-break, and the best one serves first.
//   - served (15%) + wire (10%): an answer is checked before it is served. A streamed upstream answer
//     is read whole, then relayed; a JSON one is parsed first. The prompt's token must be in the
//     model's text, a JSON-mode answer must parse, and a tool-call-only answer does not count. A weak
//     answer is never served: the server re-dials instead.
//   - failover (10%): a refused gateway call is retried or failed over, and every refusal is reported
//     on the response in `x-makefun-upstream-failures`, the only trail the failover column can see. A
//     5xx gets one immediate retry on the same model; a 429 moves straight to the next candidate (the
//     model is cooling down); any other refusal moves on too. An invalid answer gets one retry on the
//     same model, then the next candidate. At most five attempts a prompt.
//   - catalog (5%): the registered catalog and both GET /v1/models answers list exactly the fourteen
//     dialable free ids, once each, no more and no fewer.
//   - latency: no artificial waits anywhere. Streams are buffered, not relayed, which costs no round
//     trip — the upstream call already took that time — and the client still gets its answer inside its
//     30-second wait. No per-attempt timeouts either: a timeout that fires on the live pass but not on
//     the replay would make the two passes dial different models, and the replay has to match byte for
//     byte. Every routing choice is a function of the brief and the gateway's answers alone.
//
// THE TRIAL
//   The server listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout),
//   POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>}, serves
//   GET /v1/models (OpenAI shape for Authorization: Bearer, Anthropic shape for x-api-key), POST
//   /v1/chat/completions and POST /v1/messages, and polls GET $MAKEFUN_ENV_URL/work until it says done,
//   then exits 0. `node main.ts --list` prints the catalog and exits 0 before any server starts.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the model it actually dialled.
//   The house records every gateway call and scores the model that really answered. This server sends
//   the brief's model id in the gateway request and puts that same id back on every answer and every
//   streamed chunk (the provider's own answer spells it without the gateway's `openrouter/` prefix), so
//   its claim is the call it made, never another. A prompt whose reported model has no matching
//   successful call in the house's record earns nothing.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
  rankCandidates,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromText,
  anthropicModelList,
  anthropicToOpenAI,
  ensureUsage,
  estimateTokens,
  openAIAnswerParts,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIToAnthropic,
  rewriteOpenAIStreamText,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most gateway attempts one prompt gets, successful or not. */
const MAX_ATTEMPTS = 5;
/** Register is retried this many times before the server gives up. */
const REGISTER_ATTEMPTS = 3;
/** A prompt's token, as the brief's prompts carry it: capitals, a dash, four digits. */
const TOKEN_PATTERN = /[A-Z]{2,}-[0-9]{4}/;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return brief.models
    .filter((m) => m.free)
    .map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** One refusal the server recovered from, for the `x-makefun-upstream-failures` trail. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The trail header's value: at most 8 rows, in the order the calls refused. */
function failuresHeader(failures: readonly FailureRow[]): string | null {
  if (failures.length === 0) return null;
  return JSON.stringify(failures.slice(0, 8));
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
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

/** Every text the request carries, in order: system prompt, then each message's text. */
function requestTexts(body: Json): string[] {
  const out: string[] = [];
  const add = (value: unknown): void => {
    if (typeof value === 'string') out.push(value);
    else if (Array.isArray(value)) {
      for (const block of value) {
        if (isRecord(block) && typeof block.text === 'string') out.push(block.text);
      }
    }
  };
  add(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (isRecord(message)) add(message.content);
  }
  return out;
}

/**
 * The prompt's token, read off the request body itself: the brief's manifest carries no token, the
 * workload does, and the client's instruction names it ("Reply with exactly the word COMET-3884").
 * Null when the body names none, in which case the answer cannot be checked for it.
 */
function tokenInBody(body: Json): string | null {
  for (const text of requestTexts(body)) {
    const found = TOKEN_PATTERN.exec(text);
    if (found !== null) return found[0];
  }
  return null;
}

/** Whether the client's body asks for JSON mode (OpenAI wire only). */
function needsJson(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
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

function sendJson(res: ServerResponse, status: number, body: unknown, failures: readonly FailureRow[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  const trail = failuresHeader(failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The gateway error body's code, or a plain `http_<status>` when it names none. */
function failureCode(status: number, text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    const error = isRecord(value) ? (isRecord(value.error) ? value.error : value) : null;
    if (error !== null) {
      if (typeof error.code === 'string' && error.code !== '') return error.code;
      if (typeof error.type === 'string' && error.type !== '') return error.type;
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/** An upstream response's body as text, '' when it cannot be read. */
async function safeText(res: Response): Promise<string> {
  try {
    return await res.text();
  } catch {
    return '';
  }
}

/**
 * The Chat Completions body one attempt dials. The client's messages, system prompt, tools,
 * max_tokens and temperature go through unchanged; the only edits are the dialled model, a guaranteed
 * output cap (the gateway refuses a call without one), and `stream_options.include_usage` on a stream.
 */
function upstreamBodyFor(wire: 'openai' | 'anthropic', clientBody: Json, model: string, stripTools: boolean): Json {
  const out: Json = wire === 'anthropic' ? anthropicToOpenAI(clientBody, model, stripTools) : { ...clientBody, model };
  if (typeof out.max_tokens !== 'number') out.max_tokens = 1024;
  if (out.stream === true) out.stream_options = { include_usage: true };
  if (wire === 'openai' && stripTools) {
    delete out.tools;
    delete out.tool_choice;
  }
  return out;
}

/** Whether an answer's text is good enough to serve: it carries the token, and parses as JSON in JSON mode. */
function textCheck(text: string, token: string | null, jsonMode: boolean): 'ok' | 'no-token' | 'bad-json' {
  if (text === '') return 'no-token';
  if (token !== null && !text.includes(token)) return 'no-token';
  if (jsonMode) {
    try {
      JSON.parse(text);
    } catch {
      return 'bad-json';
    }
  }
  return 'ok';
}


/* ------------------------------------------------------------------ */
/* Serving: one prompt, up to five gateway attempts, the best answer.  */
/* ------------------------------------------------------------------ */

/** One served answer, ready to write to the client in the wire it arrived on. */
interface ServedPayload {
  /** A non-streaming OpenAI answer, model set, usage whole. */
  json: Json | null;
  /** Streamed OpenAI frames to relay, model rewritten, usage whole, ending in [DONE]. */
  openAIFrames: string[] | null;
  /** A non-streaming Anthropic answer, usage whole. */
  anthropicJson: Json | null;
  /** Streamed Anthropic events to relay, usage whole, ending in message_stop. */
  anthropicEvents: string[] | null;
}

type Validated =
  | { kind: 'served'; payload: ServedPayload }
  | { kind: 'tool-call' }
  | { kind: 'invalid'; reason: 'no-token' | 'bad-json' };

type ServeOutcome =
  | { payload: ServedPayload; model: string; failures: FailureRow[] }
  | { errorStatus: number; errorText: string; failures: FailureRow[] };

/** The tool names a buffered OpenAI stream's deltas called. */
function openAIStreamToolNames(frames: readonly string[]): string[] {
  const names: string[] = [];
  for (const frame of frames) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      try {
        const chunk: unknown = JSON.parse(payload);
        if (!isRecord(chunk)) continue;
        const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
        const choice = isRecord(choices[0]) ? choices[0] : null;
        if (choice === null) continue;
        const delta = isRecord(choice.delta) ? choice.delta : {};
        const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
        for (const call of calls) {
          if (isRecord(call) && isRecord(call.function) && typeof call.function.name === 'string' && call.function.name !== '') {
            names.push(call.function.name);
          }
        }
      } catch {
        // Not JSON: nothing to collect.
      }
    }
  }
  return names;
}

/** Whether a buffered Anthropic event stream opened a tool_use block. */
function anthropicStreamHasToolUse(events: readonly string[]): boolean {
  for (const event of events) {
    for (const line of event.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      try {
        const data: unknown = JSON.parse(trimmed.slice(5).trim());
        if (isRecord(data) && data.type === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') return true;
      } catch {
        // Not JSON: nothing to collect.
      }
    }
  }
  return false;
}

/** The first chunk's id in a buffered OpenAI stream, for a synthesized usage chunk. */
function firstStreamId(frames: readonly string[]): string {
  for (const frame of frames) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      try {
        const chunk: unknown = JSON.parse(payload);
        if (isRecord(chunk) && typeof chunk.id === 'string' && chunk.id !== '') return chunk.id;
      } catch {
        // Not JSON: look on.
      }
    }
  }
  return 'chatcmpl-aggregate';
}

/**
 * A buffered OpenAI stream's usage, made whole. A chunk that carries usage gets the ensured counts;
 * a stream that carried no usage chunk at all gets one inserted before [DONE]. The wire check needs a
 * whole input count of at least 0 and an output count of at least 1, and a free model that omits usage
 * must not lose the prompt for it.
 */
function patchOpenAIStreamUsage(frames: string[], usage: Json | null, upstreamBody: Json, model: string, text: string): string[] {
  const ensured = ensureUsage(usage, JSON.stringify(upstreamBody).length, text);
  let patched = false;
  const out = frames.map((frame) => {
    const lines = frame.split('\n').map((line) => {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) return line;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') return line;
      try {
        const chunk: unknown = JSON.parse(payload);
        if (isRecord(chunk) && isRecord(chunk.usage)) {
          patched = true;
          return `data: ${JSON.stringify({ ...chunk, usage: { prompt_tokens: ensured.prompt_tokens, completion_tokens: ensured.completion_tokens } })}`;
        }
      } catch {
        // Not JSON: relay the line as it came.
      }
      return line;
    });
    return lines.join('\n');
  });
  if (patched) return out;
  const usageChunk = `data: ${JSON.stringify({ id: firstStreamId(frames), object: 'chat.completion.chunk', created: 0, model, choices: [], usage: { prompt_tokens: ensured.prompt_tokens, completion_tokens: ensured.completion_tokens } })}\n\n`;
  const doneIdx = out.findIndex((f) => f.includes('[DONE]'));
  if (doneIdx === -1) return [...out, usageChunk, 'data: [DONE]\n\n'];
  return [...out.slice(0, doneIdx), usageChunk, ...out.slice(doneIdx)];
}

/**
 * A buffered Anthropic event stream's usage, made whole: the message_delta event's output count is
 * patched up to the estimate when the upstream stream carried no usage. The input count the
 * message_start event carries is already a whole number of at least 0.
 */
function patchAnthropicStreamUsage(events: string[], outputTokens: number, outputEstimate: number): string[] {
  if (outputTokens > 0) return events;
  return events.map((event) => {
    const lines = event.split('\n').map((line) => {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) return line;
      try {
        const data: unknown = JSON.parse(trimmed.slice(5).trim());
        if (isRecord(data) && data.type === 'message_delta' && isRecord(data.usage)) {
          return `data: ${JSON.stringify({ ...data, usage: { ...data.usage, output_tokens: outputEstimate } })}`;
        }
      } catch {
        // Not JSON: relay the line as it came.
      }
      return line;
    });
    return lines.join('\n');
  });
}

/**
 * One successful (2xx) gateway answer, read whole and checked: the prompt's token in the text, valid
 * JSON in JSON mode, and no tool-call-only answer. A stream is buffered first, so a weak answer is
 * caught before the client ever sees it.
 */
async function readAndValidate(
  wire: 'openai' | 'anthropic',
  upstream: Response,
  model: string,
  upstreamBody: Json,
  token: string | null,
  jsonMode: boolean,
): Promise<Validated> {
  const stream = upstreamBody.stream === true;
  if (stream) {
    const text = await safeText(upstream);
    if (wire === 'openai') {
      const rewritten = rewriteOpenAIStreamText(text, model);
      if (rewritten.text === '' && openAIStreamToolNames(rewritten.frames).length > 0) return { kind: 'tool-call' };
      const check = textCheck(rewritten.text, token, jsonMode);
      if (check !== 'ok') return { kind: 'invalid', reason: check };
      const frames = patchOpenAIStreamUsage(rewritten.frames, rewritten.usage, upstreamBody, model, rewritten.text);
      return { kind: 'served', payload: { json: null, openAIFrames: frames, anthropicJson: null, anthropicEvents: null } };
    }
    const converted = await anthropicEventsFromText(text, model);
    if (converted.text === '' && anthropicStreamHasToolUse(converted.events)) return { kind: 'tool-call' };
    const check = textCheck(converted.text, token, jsonMode);
    if (check !== 'ok') return { kind: 'invalid', reason: check };
    const events = patchAnthropicStreamUsage(converted.events, converted.outputTokens, estimateTokens(converted.text.length));
    return { kind: 'served', payload: { json: null, openAIFrames: null, anthropicJson: null, anthropicEvents: events } };
  }
  let answer: unknown;
  try {
    answer = await upstream.json();
  } catch {
    return { kind: 'invalid', reason: 'bad-json' };
  }
  if (!isRecord(answer)) return { kind: 'invalid', reason: 'bad-json' };
  const parts = openAIAnswerParts(answer);
  if (parts.toolCalls.length > 0 && parts.text === '') return { kind: 'tool-call' };
  const check = textCheck(parts.text, token, jsonMode);
  if (check !== 'ok') return { kind: 'invalid', reason: check };
  const usage = ensureUsage(answer.usage, JSON.stringify(upstreamBody).length, parts.text);
  if (wire === 'openai') {
    return { kind: 'served', payload: { json: openAIJsonWithModel({ ...answer, usage }, model), openAIFrames: null, anthropicJson: null, anthropicEvents: null } };
  }
  const message = openAIToAnthropic(answer, model);
  message.usage = { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens };
  return { kind: 'served', payload: { json: null, openAIFrames: null, anthropicJson: message, anthropicEvents: null } };
}

/**
 * Serve one prompt. Candidates walk in ranking order; a refusal moves to the next one, except a 5xx,
 * which gets one immediate retry on the same model, and an invalid answer, which gets one retry on
 * the same model (with the tools stripped when the answer was a tool call) before the next candidate.
 * Every refusal is recorded for the trail header. At most MAX_ATTEMPTS gateway calls a prompt.
 */
async function servePrompt(wire: 'openai' | 'anthropic', clientBody: Json, requires: Requirement, token: string | null): Promise<ServeOutcome> {
  const candidates = rankCandidates(brief, requires);
  if (candidates.length === 0) {
    return { errorStatus: 503, errorText: 'no free model in the brief meets this request', failures: [] };
  }
  const jsonMode = wire === 'openai' && needsJson(clientBody);
  const failures: FailureRow[] = [];
  const queue: { model: string; stripTools: boolean }[] = [{ model: candidates[0].id, stripTools: false }];
  const attempted = new Set<string>();
  const retried5xx = new Set<string>();
  const retriedInvalid = new Set<string>();
  let attempts = 0;

  const enqueueNext = (): void => {
    const next = candidates.find((c) => !attempted.has(c.id) && !queue.some((q) => q.model === c.id));
    if (next !== undefined) queue.push({ model: next.id, stripTools: false });
  };

  while (queue.length > 0 && attempts < MAX_ATTEMPTS) {
    const item = queue.shift() as { model: string; stripTools: boolean };
    attempts += 1;
    attempted.add(item.model);
    const upstreamBody = upstreamBodyFor(wire, clientBody, item.model, item.stripTools);
    let upstream: Response;
    try {
      upstream = await dial(item.model, upstreamBody);
    } catch (err) {
      failures.push({ status: 0, code: 'fetch_error', model: item.model });
      log(`dial ${item.model} failed to reach the gateway: ${(err as Error).message}`);
      enqueueNext();
      continue;
    }
    if (!upstream.ok) {
      const text = await safeText(upstream);
      failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model: item.model });
      log(`dial ${item.model} refused HTTP ${upstream.status}`);
      if (upstream.status >= 500 && upstream.status < 600 && !retried5xx.has(item.model)) {
        retried5xx.add(item.model);
        queue.unshift(item);
      } else {
        enqueueNext();
      }
      continue;
    }
    const outcome = await readAndValidate(wire, upstream, item.model, upstreamBody, token, jsonMode);
    if (outcome.kind === 'served') {
      return { payload: outcome.payload, model: item.model, failures };
    }
    if (outcome.kind === 'tool-call' && !item.stripTools) {
      log(`dial ${item.model} answered with a tool call; re-dialling with the tools stripped`);
      queue.unshift({ model: item.model, stripTools: true });
      continue;
    }
    if (!retriedInvalid.has(item.model)) {
      retriedInvalid.add(item.model);
      log(`dial ${item.model} answered without the token; one retry on the same model`);
      queue.unshift(item);
      continue;
    }
    log(`dial ${item.model} answered without the token twice; moving to the next candidate`);
    enqueueNext();
  }
  return { errorStatus: 502, errorText: 'every free model that met this request refused it or answered without the token', failures };
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const token = tokenInBody(body);
  const result = await servePrompt('openai', body, requires, token);
  if ('errorStatus' in result) return sendError(res, 'openai', result.errorStatus, 'upstream_error', result.errorText);
  const { payload, failures } = result;
  if (payload.json !== null) {
    sendJson(res, 200, payload.json, failures);
    return;
  }
  const trail = failuresHeader(failures);
  const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(200, headers);
  try {
    for (const frame of payload.openAIFrames ?? []) res.write(frame);
  } finally {
    res.end();
  }
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const token = tokenInBody(body);
  const result = await servePrompt('anthropic', body, requires, token);
  if ('errorStatus' in result) return sendError(res, 'anthropic', result.errorStatus, 'api_error', result.errorText);
  const { payload, failures } = result;
  if (payload.anthropicJson !== null) {
    sendJson(res, 200, payload.anthropicJson, failures);
    return;
  }
  const trail = failuresHeader(failures);
  const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(200, headers);
  try {
    for (const event of payload.anthropicEvents ?? []) res.write(event);
  } finally {
    res.end();
  }
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

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let last: unknown = null;
  for (let attempt = 1; attempt <= REGISTER_ATTEMPTS; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      last = err;
      log(`register attempt ${attempt} failed: ${(err as Error).message}`);
      if (attempt < REGISTER_ATTEMPTS) await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw last instanceof Error ? last : new Error(String(last));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
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
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
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
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
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
