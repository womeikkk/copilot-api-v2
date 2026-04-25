/* eslint-disable max-lines */

import { randomUUID } from "node:crypto"

import type {
  AnthropicAssistantContentBlock,
  AnthropicMessage,
  AnthropicMessagesPayload,
  AnthropicResponse,
  AnthropicStreamEventData,
  AnthropicTool,
  AnthropicToolResultContentBlock,
  AnthropicUserContentBlock,
} from "~/routes/messages/anthropic-types"
import type {
  FunctionTool,
  ResponseInputContent,
  ResponseInputFile,
  ResponseInputImage,
  ResponseInputItem,
  ResponseInputMessage,
  ResponseInputReasoning,
  ResponseOutputItem,
  ResponseStreamEvent,
  ResponsesPayload,
  ResponsesResult,
  Tool,
  ToolChoiceFunction,
  ToolChoiceOptions,
} from "~/services/copilot/create-responses"

const DEFAULT_MAX_OUTPUT_TOKENS = 64_000
const THINKING_TEXT = "Thinking..."

type ResponseCompletionState = {
  status: ResponsesResult["status"]
  incompleteDetails: ResponsesResult["incomplete_details"]
}

export const translateResponsesPayloadToAnthropicMessages = (
  payload: ResponsesPayload,
): AnthropicMessagesPayload => {
  const messages: Array<AnthropicMessage> = []
  const systemPrompts: Array<string> = []

  if (typeof payload.instructions === "string" && payload.instructions.trim()) {
    systemPrompts.push(payload.instructions)
  }

  if (typeof payload.input === "string") {
    messages.push({ role: "user", content: payload.input })
  } else if (Array.isArray(payload.input)) {
    for (const item of payload.input) {
      translateResponsesInputItem(item, messages, systemPrompts)
    }
  }

  return {
    model: payload.model,
    messages,
    max_tokens: payload.max_output_tokens ?? DEFAULT_MAX_OUTPUT_TOKENS,
    ...(systemPrompts.length > 0 && { system: systemPrompts.join("\n\n") }),
    ...(payload.stream !== undefined && { stream: Boolean(payload.stream) }),
    ...(payload.temperature !== undefined
      && payload.temperature !== null && { temperature: payload.temperature }),
    ...(payload.top_p !== undefined
      && payload.top_p !== null && { top_p: payload.top_p }),
    ...(payload.tools && { tools: convertResponsesTools(payload.tools) }),
    ...(payload.tool_choice && {
      tool_choice: convertResponsesToolChoice(payload.tool_choice),
    }),
  }
}

const translateResponsesInputItem = (
  item: ResponseInputItem,
  messages: Array<AnthropicMessage>,
  systemPrompts: Array<string>,
): void => {
  const type = (item as { type?: unknown }).type

  if (type === "message" || isResponsesMessage(item)) {
    translateResponsesMessage(
      item as ResponseInputMessage,
      messages,
      systemPrompts,
    )
    return
  }

  if (type === "function_call") {
    const call = item as {
      call_id: string
      name: string
      arguments: string
    }
    messages.push({
      role: "assistant",
      content: [
        {
          type: "tool_use",
          id: call.call_id,
          name: call.name,
          input: parseJsonObject(call.arguments),
        },
      ],
    })
    return
  }

  if (type === "function_call_output") {
    const output = item as {
      call_id: string
      output: string | Array<ResponseInputContent>
      status?: string
    }
    messages.push({
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: output.call_id,
          content: convertFunctionCallOutput(output.output),
          ...(output.status === "incomplete" && { is_error: true }),
        },
      ],
    })
    return
  }

  if (type === "reasoning") {
    const reasoning = item as ResponseInputReasoning
    messages.push({
      role: "assistant",
      content: [
        {
          type: "thinking",
          thinking: combineReasoningSummary(reasoning.summary),
          signature: reasoning.encrypted_content,
        },
      ],
    })
  }
}

const translateResponsesMessage = (
  item: ResponseInputMessage,
  messages: Array<AnthropicMessage>,
  systemPrompts: Array<string>,
): void => {
  const content = convertResponseMessageContent(item.content)

  if (item.role === "system" || item.role === "developer") {
    const text = flattenContentToText(item.content)
    if (text) {
      systemPrompts.push(text)
    }
    return
  }

  if (item.role === "assistant") {
    messages.push({
      role: "assistant",
      content,
    })
    return
  }

  messages.push({
    role: "user",
    content,
  })
}

const convertResponseMessageContent = (
  content: ResponseInputMessage["content"],
):
  | string
  | Array<AnthropicUserContentBlock>
  | Array<AnthropicAssistantContentBlock> => {
  if (typeof content === "string") {
    return content
  }

  if (!Array.isArray(content)) {
    return ""
  }

  const blocks = content.flatMap((block) => convertResponseInputContent(block))
  return blocks.length > 0 ? blocks : ""
}

const convertResponseInputContent = (
  content: ResponseInputContent,
): Array<AnthropicUserContentBlock | AnthropicAssistantContentBlock> => {
  const type = (content as { type?: unknown }).type

  switch (type) {
    case "input_text":
    case "output_text": {
      const text = (content as { text?: unknown }).text
      return typeof text === "string" ? [{ type: "text", text }] : []
    }
    case "input_image": {
      return convertResponseImageContent(content as ResponseInputImage)
    }
    case "input_file": {
      return convertResponseFileContent(content as ResponseInputFile)
    }
    default: {
      return []
    }
  }
}

const convertResponseImageContent = (
  content: ResponseInputImage,
): Array<AnthropicUserContentBlock> => {
  const parsed = parseDataUrl(content.image_url)
  if (!parsed) {
    return []
  }

  if (!isAnthropicImageMediaType(parsed.mediaType)) {
    return []
  }

  return [
    {
      type: "image",
      source: {
        type: "base64",
        media_type: parsed.mediaType,
        data: parsed.data,
      },
    },
  ]
}

const convertResponseFileContent = (
  content: ResponseInputFile,
): Array<AnthropicUserContentBlock> => {
  const parsed = parseDataUrl(content.file_data)
  if (!parsed || parsed.mediaType !== "application/pdf") {
    return []
  }

  return [
    {
      type: "document",
      source: {
        type: "base64",
        media_type: "application/pdf",
        data: parsed.data,
      },
      title: content.filename,
    },
  ]
}

const convertFunctionCallOutput = (
  output: string | Array<ResponseInputContent>,
): string | Array<AnthropicToolResultContentBlock> => {
  if (typeof output === "string") {
    return output
  }

  if (!Array.isArray(output)) {
    return ""
  }

  const blocks = output.flatMap((content) => {
    const converted = convertResponseInputContent(content)
    return converted.filter(
      (block): block is AnthropicToolResultContentBlock =>
        block.type === "text"
        || block.type === "image"
        || block.type === "document",
    )
  })

  return blocks.length > 0 ? blocks : ""
}

const convertResponsesTools = (tools: Array<Tool>): Array<AnthropicTool> => {
  return tools.flatMap((tool) => {
    if ((tool as { type?: unknown }).type !== "function") {
      return []
    }

    const functionTool = tool as FunctionTool
    return [
      {
        name: functionTool.name,
        description: functionTool.description ?? undefined,
        input_schema: functionTool.parameters ?? { type: "object" },
      },
    ]
  })
}

const convertResponsesToolChoice = (
  toolChoice: ToolChoiceOptions | ToolChoiceFunction,
): AnthropicMessagesPayload["tool_choice"] => {
  if (toolChoice === "auto") {
    return { type: "auto" }
  }

  if (toolChoice === "required") {
    return { type: "any" }
  }

  if (toolChoice === "none") {
    return { type: "none" }
  }

  return { type: "tool", name: toolChoice.name }
}

export const translateAnthropicResponseToResponsesResult = (
  response: AnthropicResponse,
  request: ResponsesPayload,
): ResponsesResult => {
  const output = convertAnthropicContentToResponsesOutput(response)
  const outputText = output
    .flatMap((item) =>
      item.type === "message" ?
        (item.content?.flatMap((block) =>
          block.type === "output_text" ? [block.text] : [],
        ) ?? [])
      : [],
    )
    .join("")

  const usage = mapAnthropicUsageToResponses(response.usage)
  const { status, incompleteDetails } = mapAnthropicStopReason(
    response.stop_reason,
  )

  return {
    id: response.id,
    object: "response",
    created_at: Math.floor(Date.now() / 1000),
    model: response.model,
    output,
    output_text: outputText,
    status,
    usage,
    error: null,
    incomplete_details: incompleteDetails,
    instructions: request.instructions ?? null,
    metadata: request.metadata ?? null,
    parallel_tool_calls: request.parallel_tool_calls ?? true,
    temperature: request.temperature ?? null,
    tool_choice: request.tool_choice ?? "auto",
    tools: request.tools ?? [],
    top_p: request.top_p ?? null,
  }
}

const convertAnthropicContentToResponsesOutput = (
  response: AnthropicResponse,
): Array<ResponseOutputItem> => {
  return response.content.flatMap((block, index) => {
    switch (block.type) {
      case "thinking": {
        return [
          {
            id: `${response.id}_reasoning_${index}`,
            type: "reasoning",
            summary:
              block.thinking && block.thinking !== THINKING_TEXT ?
                [{ type: "summary_text", text: block.thinking }]
              : [],
            encrypted_content: block.signature,
            status: "completed",
          },
        ]
      }
      case "tool_use": {
        return [
          {
            id: `${response.id}_function_${index}`,
            type: "function_call",
            call_id: block.id,
            name: block.name,
            arguments: JSON.stringify(block.input),
            status: "completed",
          },
        ]
      }
      case "text": {
        return [
          {
            id: `${response.id}_message_${index}`,
            type: "message",
            role: "assistant",
            status: "completed",
            content: [
              { type: "output_text", text: block.text, annotations: [] },
            ],
          },
        ]
      }
      default: {
        return []
      }
    }
  })
}

const mapAnthropicUsageToResponses = (
  usage: AnthropicResponse["usage"],
): ResponsesResult["usage"] => {
  const cachedTokens = usage.cache_read_input_tokens ?? 0
  const inputTokens = usage.input_tokens + cachedTokens
  const outputTokens = usage.output_tokens

  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens,
    ...(cachedTokens > 0 && {
      input_tokens_details: {
        cached_tokens: cachedTokens,
      },
    }),
  }
}

const mapAnthropicStopReason = (
  stopReason: AnthropicResponse["stop_reason"],
): ResponseCompletionState => {
  if (stopReason === "max_tokens") {
    return {
      status: "incomplete",
      incomplete_details: { reason: "max_output_tokens" },
    }
  }

  return {
    status: "completed",
    incomplete_details: null,
  }
}

interface AnthropicResponsesStreamState {
  request: ResponsesPayload
  responseId: string
  model: string
  createdAt: number
  sequenceNumber: number
  completed: boolean
  output: Array<ResponseOutputItem>
  outputText: string
  openBlocks: Map<number, OpenAnthropicBlock>
  finalUsage?: AnthropicResponse["usage"]
  finalStopReason?: AnthropicResponse["stop_reason"]
  finalStopSequence?: string | null
}

type OpenAnthropicBlock =
  | {
      kind: "message"
      outputIndex: number
      itemId: string
      text: string
    }
  | {
      kind: "reasoning"
      outputIndex: number
      itemId: string
      text: string
      signature: string
    }
  | {
      kind: "function_call"
      outputIndex: number
      itemId: string
      callId: string
      name: string
      arguments: string
    }

export const createAnthropicResponsesStreamState = (
  request: ResponsesPayload,
): AnthropicResponsesStreamState => ({
  request,
  responseId: `resp_${randomUUID()}`,
  model: request.model,
  createdAt: Math.floor(Date.now() / 1000),
  sequenceNumber: 0,
  completed: false,
  output: [],
  outputText: "",
  openBlocks: new Map(),
})

export const translateAnthropicStreamEventToResponses = (
  rawEvent: AnthropicStreamEventData,
  state: AnthropicResponsesStreamState,
): Array<ResponseStreamEvent> => {
  switch (rawEvent.type) {
    case "ping": {
      return []
    }
    case "message_start": {
      state.responseId = rawEvent.message.id
      state.model = rawEvent.message.model
      state.finalUsage = {
        input_tokens: rawEvent.message.usage.input_tokens,
        output_tokens: rawEvent.message.usage.output_tokens,
        cache_read_input_tokens: rawEvent.message.usage.cache_read_input_tokens,
      }
      return [
        {
          type: "response.created",
          sequence_number: nextSequence(state),
          response: buildResponsesResult(state, "in_progress", null),
        },
      ]
    }
    case "content_block_start": {
      return handleAnthropicContentBlockStart(rawEvent, state)
    }
    case "content_block_delta": {
      return handleAnthropicContentBlockDelta(rawEvent, state)
    }
    case "content_block_stop": {
      return handleAnthropicContentBlockStop(rawEvent.index, state)
    }
    case "message_delta": {
      state.finalStopReason = rawEvent.delta.stop_reason
      state.finalStopSequence = rawEvent.delta.stop_sequence
      if (rawEvent.usage) {
        state.finalUsage = {
          input_tokens:
            rawEvent.usage.input_tokens ?? state.finalUsage?.input_tokens ?? 0,
          output_tokens: rawEvent.usage.output_tokens,
          cache_read_input_tokens: rawEvent.usage.cache_read_input_tokens,
        }
      }
      return []
    }
    case "message_stop": {
      const completed = handleAllOpenAnthropicBlocks(state)
      const { status, incompleteDetails } = mapAnthropicStopReason(
        state.finalStopReason ?? "end_turn",
      )
      state.completed = true
      return [
        ...completed,
        {
          type:
            status === "completed" ? "response.completed" : (
              "response.incomplete"
            ),
          sequence_number: nextSequence(state),
          response: buildResponsesResult(state, status, incompleteDetails),
        },
      ]
    }
    case "error": {
      state.completed = true
      return [
        {
          type: "error",
          code: rawEvent.error.type,
          message: rawEvent.error.message,
          param: null,
          sequence_number: nextSequence(state),
        },
      ]
    }
    default: {
      return []
    }
  }
}

export const translateAnthropicResponsesStreamStateToResponses = (
  state: AnthropicResponsesStreamState,
): Array<ResponseStreamEvent> => {
  if (state.completed) {
    return []
  }

  const events = handleAllOpenAnthropicBlocks(state)
  const { status, incompleteDetails } = mapAnthropicStopReason(
    state.finalStopReason ?? "end_turn",
  )
  state.completed = true

  events.push({
    type: status === "completed" ? "response.completed" : "response.incomplete",
    sequence_number: nextSequence(state),
    response: buildResponsesResult(state, status, incompleteDetails),
  })

  return events
}

const handleAnthropicContentBlockStart = (
  event: Extract<AnthropicStreamEventData, { type: "content_block_start" }>,
  state: AnthropicResponsesStreamState,
): Array<ResponseStreamEvent> => {
  const outputIndex = state.output.length
  const block = event.content_block

  if (block.type === "text") {
    const itemId = `${state.responseId}_message_${outputIndex}`
    const item: ResponseOutputItem = {
      id: itemId,
      type: "message",
      role: "assistant",
      status: "in_progress",
      content: [{ type: "output_text", text: "", annotations: [] }],
    }
    state.output.push(item)
    state.openBlocks.set(event.index, {
      kind: "message",
      outputIndex,
      itemId,
      text: "",
    })
    return [
      {
        type: "response.output_item.added",
        sequence_number: nextSequence(state),
        output_index: outputIndex,
        item,
      },
    ]
  }

  if (block.type === "thinking") {
    const itemId = `${state.responseId}_reasoning_${outputIndex}`
    const item: ResponseOutputItem = {
      id: itemId,
      type: "reasoning",
      summary: [],
      encrypted_content: "",
      status: "in_progress",
    }
    state.output.push(item)
    state.openBlocks.set(event.index, {
      kind: "reasoning",
      outputIndex,
      itemId,
      text: "",
      signature: "",
    })
    return [
      {
        type: "response.output_item.added",
        sequence_number: nextSequence(state),
        output_index: outputIndex,
        item,
      },
    ]
  }

  const itemId = `${state.responseId}_function_${outputIndex}`
  const item: ResponseOutputItem = {
    id: itemId,
    type: "function_call",
    call_id: block.id,
    name: block.name,
    arguments: "",
    status: "in_progress",
  }
  state.output.push(item)
  state.openBlocks.set(event.index, {
    kind: "function_call",
    outputIndex,
    itemId,
    callId: block.id,
    name: block.name,
    arguments: "",
  })
  return [
    {
      type: "response.output_item.added",
      sequence_number: nextSequence(state),
      output_index: outputIndex,
      item,
    },
  ]
}

const handleAnthropicContentBlockDelta = (
  event: Extract<AnthropicStreamEventData, { type: "content_block_delta" }>,
  state: AnthropicResponsesStreamState,
): Array<ResponseStreamEvent> => {
  const block = state.openBlocks.get(event.index)
  if (!block) {
    return []
  }

  if (block.kind === "message" && event.delta.type === "text_delta") {
    block.text += event.delta.text
    state.outputText += event.delta.text
    return [
      {
        type: "response.output_text.delta",
        sequence_number: nextSequence(state),
        output_index: block.outputIndex,
        item_id: block.itemId,
        content_index: 0,
        delta: event.delta.text,
      },
    ]
  }

  if (block.kind === "reasoning") {
    if (event.delta.type === "thinking_delta") {
      block.text += event.delta.thinking
      return [
        {
          type: "response.reasoning_summary_text.delta",
          sequence_number: nextSequence(state),
          output_index: block.outputIndex,
          item_id: block.itemId,
          summary_index: 0,
          delta: event.delta.thinking,
        },
      ]
    }

    if (event.delta.type === "signature_delta") {
      block.signature += event.delta.signature
    }
    return []
  }

  if (
    block.kind === "function_call"
    && event.delta.type === "input_json_delta"
  ) {
    block.arguments += event.delta.partial_json
    return [
      {
        type: "response.function_call_arguments.delta",
        sequence_number: nextSequence(state),
        output_index: block.outputIndex,
        item_id: block.itemId,
        delta: event.delta.partial_json,
      },
    ]
  }

  return []
}

const handleAnthropicContentBlockStop = (
  blockIndex: number,
  state: AnthropicResponsesStreamState,
): Array<ResponseStreamEvent> => {
  const block = state.openBlocks.get(blockIndex)
  if (!block) {
    return []
  }

  state.openBlocks.delete(blockIndex)

  if (block.kind === "message") {
    const item = state.output[block.outputIndex] as Extract<
      ResponseOutputItem,
      { type: "message" }
    >
    item.status = "completed"
    item.content = [
      {
        type: "output_text",
        text: block.text,
        annotations: [],
      },
    ]
    return [
      {
        type: "response.output_text.done",
        sequence_number: nextSequence(state),
        output_index: block.outputIndex,
        item_id: block.itemId,
        content_index: 0,
        text: block.text,
      },
      {
        type: "response.output_item.done",
        sequence_number: nextSequence(state),
        output_index: block.outputIndex,
        item,
      },
    ]
  }

  if (block.kind === "reasoning") {
    const item = state.output[block.outputIndex] as Extract<
      ResponseOutputItem,
      { type: "reasoning" }
    >
    item.status = "completed"
    item.summary =
      block.text && block.text !== THINKING_TEXT ?
        [{ type: "summary_text", text: block.text }]
      : []
    item.encrypted_content = block.signature

    const events: Array<ResponseStreamEvent> = []
    if (block.text) {
      events.push({
        type: "response.reasoning_summary_text.done",
        sequence_number: nextSequence(state),
        output_index: block.outputIndex,
        item_id: block.itemId,
        summary_index: 0,
        text: block.text,
      })
    }
    events.push({
      type: "response.output_item.done",
      sequence_number: nextSequence(state),
      output_index: block.outputIndex,
      item,
    })
    return events
  }

  const item = state.output[block.outputIndex] as Extract<
    ResponseOutputItem,
    { type: "function_call" }
  >
  item.status = "completed"
  item.arguments = block.arguments
  return [
    {
      type: "response.function_call_arguments.done",
      sequence_number: nextSequence(state),
      output_index: block.outputIndex,
      item_id: block.itemId,
      name: block.name,
      arguments: block.arguments,
    },
    {
      type: "response.output_item.done",
      sequence_number: nextSequence(state),
      output_index: block.outputIndex,
      item,
    },
  ]
}

const handleAllOpenAnthropicBlocks = (
  state: AnthropicResponsesStreamState,
): Array<ResponseStreamEvent> => {
  const events: Array<ResponseStreamEvent> = []
  const openIndices = [...state.openBlocks.keys()].sort(
    (left, right) => left - right,
  )

  for (const blockIndex of openIndices) {
    events.push(...handleAnthropicContentBlockStop(blockIndex, state))
  }

  return events
}

const buildResponsesResult = (
  state: AnthropicResponsesStreamState,
  status: ResponsesResult["status"],
  incompleteDetails: ResponsesResult["incomplete_details"],
): ResponsesResult => {
  const usage =
    state.finalUsage ? mapAnthropicUsageToResponses(state.finalUsage) : null

  return {
    id: state.responseId,
    object: "response",
    created_at: state.createdAt,
    model: state.model,
    output: [...state.output],
    output_text: state.outputText,
    status,
    usage,
    error: null,
    incomplete_details: incompleteDetails,
    instructions: state.request.instructions ?? null,
    metadata: state.request.metadata ?? null,
    parallel_tool_calls: state.request.parallel_tool_calls ?? true,
    temperature: state.request.temperature ?? null,
    tool_choice: state.request.tool_choice ?? "auto",
    tools: state.request.tools ?? [],
    top_p: state.request.top_p ?? null,
  }
}

const combineReasoningSummary = (
  summary: ResponseInputReasoning["summary"],
): string => {
  const text = summary.map((block) => block.text).join("")
  return text || THINKING_TEXT
}

const flattenContentToText = (
  content: ResponseInputMessage["content"],
): string => {
  if (typeof content === "string") {
    return content
  }

  if (!Array.isArray(content)) {
    return ""
  }

  return content
    .flatMap((block) => {
      const text = (block as { text?: unknown }).text
      return typeof text === "string" ? [text] : []
    })
    .join("\n\n")
}

const parseJsonObject = (rawArguments: string): Record<string, unknown> => {
  if (typeof rawArguments !== "string" || rawArguments.trim() === "") {
    return {}
  }

  try {
    const parsed = JSON.parse(rawArguments) as unknown
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
  } catch {
    return { raw_arguments: rawArguments }
  }

  return { arguments: rawArguments }
}

const parseDataUrl = (
  value: string | null | undefined,
): { mediaType: string; data: string } | null => {
  if (!value) {
    return null
  }

  const match = value.match(/^data:([^;]+);base64,(.+)$/u)
  if (!match) {
    return null
  }

  return {
    mediaType: match[1],
    data: match[2],
  }
}

const isResponsesMessage = (item: ResponseInputItem): boolean => {
  return typeof (item as { role?: unknown }).role === "string"
}

const isAnthropicImageMediaType = (
  mediaType: string,
): mediaType is "image/jpeg" | "image/png" | "image/gif" | "image/webp" => {
  return ["image/gif", "image/jpeg", "image/png", "image/webp"].includes(
    mediaType,
  )
}

const nextSequence = (state: AnthropicResponsesStreamState): number => {
  state.sequenceNumber += 1
  return state.sequenceNumber
}
