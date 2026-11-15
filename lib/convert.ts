// Wire conversions for the aggregator: requests in, answers out.
//
// The client compares what a provider echoed against what the client sent, so a conversion may never
// drop or rewrite a field the echo reports: the system prompt, the last user text, the images, the tool
// names, max_tokens, the stop sequences, or JSON mode.
//
// Answers are normalised first (one internal shape per provider wire) and then emitted in the client's
// wire, streamed or not. Every upstream call is non-streaming: a provider's own event stream may or may
// not carry usage, and the client checks usage, so the entry asks for the whole answer and synthesises
// the client's event stream itself.

import { type Json, isRecord } from './brief.ts';
import { anthropicToOpenAI, sseEvent } from './wires.ts';

const textOf = (value: unknown): string => {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return '';
  const parts: string[] = [];
  for (const block of value) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('\n');
};

const num = (value: unknown): number | undefined => (typeof value === 'number' && Number.isFinite(value) ? value : undefined);
const strList = (value: unknown): string[] => {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  const out: string[] = [];
  for (const item of value) if (typeof item === 'string') out.push(item);
  return out;
};

/** A data: URL as the two fields Gemini's inlineData wants. */
function inlineOf(url: unknown): Json | null {
  if (typeof url !== 'string') return null;
  const data = /^data:([^;,]*)?;base64,(.*)$/s.exec(url);
  if (data === null) return { fileData: { fileUri: url } };
  return { inlineData: { mimeType: data[1] === '' ? 'image/png' : data[1], data: data[2] } };
}

const jsonModeOf = (source: Json): boolean => {
  const format = isRecord(source.response_format) ? source.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
};

/** OpenAI tools (or Anthropic tools) as Gemini functionDeclarations. */
function geminiTools(source: Json): Json[] | undefined {
  const tools = Array.isArray(source.tools) ? source.tools : [];
  const declarations: Json[] = [];
  for (const tool of tools) {
    if (!isRecord(tool)) continue;
    if (isRecord(tool.function)) {
      const fn: Json = { name: tool.function.name };
      if (typeof tool.function.description === 'string') fn.description = tool.function.description;
      if (isRecord(tool.function.parameters)) fn.parameters = tool.function.parameters;
      declarations.push(fn);
    } else if (typeof tool.name === 'string') {
      const fn: Json = { name: tool.name };
      if (typeof tool.description === 'string') fn.description = tool.description;
      if (isRecord(tool.input_schema)) fn.parameters = tool.input_schema;
      declarations.push(fn);
    }
  }
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : undefined;
}

/** An OpenAI Chat Completions request as a Gemini generateContent request. */
export function openAIToGemini(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  for (const message of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(message)) continue;
    const role = message.role;
    if (role === 'system' || role === 'developer') {
      const text = textOf(message.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const parts: Json[] = [];
    if (typeof message.content === 'string') parts.push({ text: message.content });
    else if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url)) {
          const inline = inlineOf(block.image_url.url);
          if (inline !== null) parts.push(inline);
        }
      }
    }
    if (parts.length === 0) parts.push({ text: '' });
    contents.push({ role: role === 'assistant' ? 'model' : 'user', parts });
  }
  if (contents.length === 0) contents.push({ role: 'user', parts: [{ text: '' }] });
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const tools = geminiTools(body);
  if (tools !== undefined) out.tools = tools;
  const config: Json = {};
  const max = num(body.max_tokens) ?? num(body.max_completion_tokens);
  if (max !== undefined) config.maxOutputTokens = max;
  const temperature = num(body.temperature);
  if (temperature !== undefined) config.temperature = temperature;
  const topP = num(body.top_p);
  if (topP !== undefined) config.topP = topP;
  const stops = strList(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonModeOf(body)) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent request. */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  for (const message of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(message)) continue;
    const parts: Json[] = [];
    const content = message.content;
    if (typeof content === 'string') parts.push({ text: content });
    else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          const source = block.source;
          if (source.type === 'base64' && typeof source.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
          } else if (source.type === 'url' && typeof source.url === 'string') {
            const inline = inlineOf(source.url);
            if (inline !== null) parts.push(inline);
          }
        } else if (block.type === 'tool_result') parts.push({ text: textOf(block.content) });
      }
    }
    if (parts.length === 0) parts.push({ text: '' });
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  if (contents.length === 0) contents.push({ role: 'user', parts: [{ text: '' }] });
  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = geminiTools(body);
  if (tools !== undefined) out.tools = tools;
  const config: Json = {};
  const max = num(body.max_tokens);
  if (max !== undefined) config.maxOutputTokens = max;
  const temperature = num(body.temperature);
  if (temperature !== undefined) config.temperature = temperature;
  const topP = num(body.top_p);
  if (topP !== undefined) config.topP = topP;
  const stops = strList(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An OpenAI Chat Completions request as an Anthropic Messages request. */
export function openAIToAnthropicRequest(body: Json): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  for (const message of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(message)) continue;
    if (message.role === 'system' || message.role === 'developer') {
      const text = textOf(message.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const blocks: Json[] = [];
    if (typeof message.content === 'string') blocks.push({ type: 'text', text: message.content });
    else if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const data = /^data:([^;,]*)?;base64,(.*)$/s.exec(block.image_url.url);
          if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1], data: data[2] } });
          else blocks.push({ type: 'image', source: { type: 'url', url: block.image_url.url } });
        }
      }
    }
    if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
    messages.push({ role: message.role === 'assistant' ? 'assistant' : 'user', content: blocks });
  }
  const out: Json = { messages, max_tokens: num(body.max_tokens) ?? 1024 };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const temperature = num(body.temperature);
  if (temperature !== undefined) out.temperature = temperature;
  const topP = num(body.top_p);
  if (topP !== undefined) out.top_p = topP;
  const stops = strList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  const declared: Json[] = [];
  const tools = Array.isArray(body.tools) ? body.tools : [];
  for (const tool of tools) {
    if (isRecord(tool) && isRecord(tool.function)) {
      const fn: Json = { name: tool.function.name };
      if (typeof tool.function.description === 'string') fn.description = tool.function.description;
      fn.input_schema = isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} };
      declared.push(fn);
    } else if (isRecord(tool) && typeof tool.name === 'string') {
      // Already an Anthropic tool: keep it as it is.
      declared.push(tool);
    }
  }
  if (declared.length > 0) out.tools = declared;
  return out;
}

// ---------------------------------------------------------------------------------------------
// Answers, normalised.
// ---------------------------------------------------------------------------------------------

export type Finish = 'stop' | 'length' | 'tool_calls' | 'content_filter';

export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface Answer {
  text: string;
  toolCalls: ToolCall[];
  input: number;
  output: number;
  finish: Finish;
}

const finishOf = (reason: unknown): Finish => {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 'length';
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 'tool_calls';
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY' || reason === 'RECITATION') return 'content_filter';
  return 'stop';
};

export function answerFromOpenAI(body: Json): Answer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: ToolCall[] = [];
  for (const call of Array.isArray(message.tool_calls) ? message.tool_calls : []) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : `call_${toolCalls.length}`,
      name: typeof call.function.name === 'string' ? call.function.name : '',
      arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    input: num(usage.prompt_tokens) ?? 0,
    output: num(usage.completion_tokens) ?? 0,
    finish: toolCalls.length > 0 ? 'tool_calls' : finishOf(choice.finish_reason),
  };
}

export function answerFromAnthropic(body: Json): Answer {
  let text = '';
  const toolCalls: ToolCall[] = [];
  for (const block of Array.isArray(body.content) ? body.content : []) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : `call_${toolCalls.length}`,
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(block.input ?? {}),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    input: num(usage.input_tokens) ?? 0,
    output: num(usage.output_tokens) ?? 0,
    finish: toolCalls.length > 0 ? 'tool_calls' : finishOf(body.stop_reason),
  };
}

export function answerFromGemini(body: Json): Answer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: ToolCall[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        arguments: JSON.stringify(part.functionCall.args ?? {}),
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    toolCalls,
    input: num(usage.promptTokenCount) ?? 0,
    output: num(usage.candidatesTokenCount) ?? 0,
    finish: toolCalls.length > 0 ? 'tool_calls' : finishOf(first.finishReason),
  };
}

// ---------------------------------------------------------------------------------------------
// Emitters: one normalised answer, in the wire the client speaks.
// ---------------------------------------------------------------------------------------------

export function openAIJson(answer: Answer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call) => ({ id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments } }));
  }
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: 0,
    model,
    choices: [{ index: 0, message, finish_reason: answer.finish, logprobs: null }],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

export function anthropicJson(answer: Answer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.arguments);
    } catch {
      input = {};
    }
    content.push({ type: 'tool_use', id: call.id, name: call.name, input });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: answer.finish === 'length' ? 'max_tokens' : answer.finish === 'tool_calls' ? 'tool_use' : answer.finish === 'content_filter' ? 'refusal' : 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

/** The whole answer as an OpenAI event stream, in the order the client checks. */
export function openAISse(answer: Answer, model: string): string {
  const chunk = (delta: Json, finish: string | null, usage?: Json): string =>
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: 0,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
      ...(usage === undefined ? {} : { usage }),
    });
  let out = chunk({ role: 'assistant', content: '' }, null);
  for (const call of answer.toolCalls) {
    out += chunk({ tool_calls: [{ index: answer.toolCalls.indexOf(call), id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments } }] }, null);
  }
  if (answer.text !== '') {
    // Split on the same boundaries the provider would: a few characters a chunk keeps the order visible.
    for (let i = 0; i < answer.text.length; i += 24) out += chunk({ content: answer.text.slice(i, i + 24) }, null);
  }
  out += chunk({}, answer.finish, { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output });
  out += 'data: [DONE]\n\n';
  return out;
}

/** The whole answer as an Anthropic event stream, in the order the client checks. */
export function anthropicSse(answer: Answer, model: string): string {
  let out = sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_aggregate',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: answer.input, output_tokens: 0 },
    },
  });
  let index = 0;
  if (answer.text !== '') {
    out += sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
    for (let i = 0; i < answer.text.length; i += 24) {
      out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: answer.text.slice(i, i + 24) } });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  for (const call of answer.toolCalls) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.arguments);
    } catch {
      input = {};
    }
    out += sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} } });
    out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: JSON.stringify(input) } });
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  if (index === 0) {
    out += sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  }
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: {
      stop_reason: answer.finish === 'length' ? 'max_tokens' : answer.finish === 'tool_calls' ? 'tool_use' : answer.finish === 'content_filter' ? 'refusal' : 'end_turn',
      stop_sequence: null,
    },
    usage: { output_tokens: answer.output },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}

export { anthropicToOpenAI, textOf };
