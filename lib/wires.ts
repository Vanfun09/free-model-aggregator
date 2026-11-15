// The wire shapes this server answers clients in.
//
// A client speaks one of two wires and its auth header says which: `Authorization: *** is an OpenAI
// Chat Completions client, `x-api-key: *** an Anthropic Messages client. Both are answered in the shape
// they asked for, and the two model lists below match whatever the entry registered.

import { type Json, isRecord } from './brief.ts';

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string | null;
}

/** GET /v1/models for a client that sent an Authorization header. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((model) => ({ id: model.id, object: 'model', created: 0, owned_by: model.provider })),
  };
}

/** GET /v1/models for a client that sent an x-api-key header. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((model) => ({ type: 'model', id: model.id, display_name: model.display_name ?? model.id })),
    has_more: false,
  };
}

/** An error in the Anthropic envelope. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in the OpenAI envelope. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, param: null, code } };
}

/** One server-sent event: an optional `event:` line, the data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

const blocksText = (content: unknown): string => {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('\n');
};

/**
 * An Anthropic Messages request as the OpenAI Chat Completions request the same answer can be asked in:
 * the system prompt becomes a system message, text and image blocks become their OpenAI counterparts,
 * tool_use blocks become tool_calls, tool_result blocks become tool messages, and the output cap, the
 * stop sequences, the tools and the sampling fields carry over under their OpenAI names.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = blocksText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    const toolCalls: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
          blocks.push({ type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } });
        } else if (source.type === 'url' && typeof source.url === 'string') {
          blocks.push({ type: 'image_url', image_url: { url: source.url } });
        }
      } else if (block.type === 'tool_use') {
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : 'call_0',
          type: 'function',
          function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
        });
      } else if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : 'call_0', content: blocksText(block.content) });
      }
    }
    if (blocks.length === 0 && toolCalls.length === 0) blocks.push({ type: 'text', text: '' });
    if (toolCalls.length > 0) {
      const message: Json = { role, content: blocksText(blocks) === '' ? null : blocksText(blocks), tool_calls: toolCalls };
      messages.push(message);
      continue;
    }
    messages.push({ role, content: blocks });
  }
  const out: Json = { model, messages };
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  for (const field of ['temperature', 'top_p'] as const) {
    if (typeof body[field] === 'number') out[field] = body[field];
  }
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences : [];
  if (stops.length > 0) out.stop = stops;
  const tools: Json[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    const fn: Json = { name: raw.name };
    if (typeof raw.description === 'string') fn.description = raw.description;
    fn.parameters = isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} };
    tools.push({ type: 'function', function: fn });
  }
  if (tools.length > 0) out.tools = tools;
  if (isRecord(body.tool_choice) && typeof body.tool_choice.type === 'string') {
    if (body.tool_choice.type === 'auto') out.tool_choice = 'auto';
    else if (body.tool_choice.type === 'any') out.tool_choice = 'required';
    else if (body.tool_choice.type === 'none') out.tool_choice = 'none';
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: body.tool_choice.name } };
    }
  }
  return out;
}
