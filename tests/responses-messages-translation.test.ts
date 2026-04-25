import { describe, expect, test } from "bun:test"

import type {
  AnthropicResponse,
  AnthropicStreamEventData,
} from "~/routes/messages/anthropic-types"
import type { ResponsesPayload } from "~/services/copilot/create-responses"

import {
  createAnthropicResponsesStreamState,
  translateAnthropicResponseToResponsesResult,
  translateAnthropicResponsesStreamStateToResponses,
  translateAnthropicStreamEventToResponses,
  translateResponsesPayloadToAnthropicMessages,
} from "~/routes/responses/messages-translation"

describe("responses/messages translation", () => {
  test("translates responses payloads into anthropic messages payloads", () => {
    const payload: ResponsesPayload = {
      model: "claude-opus-4.6",
      instructions: "follow the instructions",
      max_output_tokens: 2048,
      input: [
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "hello" }],
        },
        {
          type: "function_call",
          call_id: "call_1",
          name: "TodoWrite",
          arguments: '{"todos":[]}',
        },
        {
          type: "function_call_output",
          call_id: "call_1",
          output: "done",
        },
      ],
      tools: [
        {
          type: "function",
          name: "TodoWrite",
          strict: false,
          parameters: { type: "object" },
        },
      ],
      tool_choice: "required",
    }

    const translated = translateResponsesPayloadToAnthropicMessages(payload)

    expect(translated).toEqual({
      model: "claude-opus-4.6",
      system: "follow the instructions",
      max_tokens: 2048,
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "hello" }],
        },
        {
          role: "assistant",
          content: [
            {
              type: "tool_use",
              id: "call_1",
              name: "TodoWrite",
              input: { todos: [] },
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "call_1",
              content: "done",
            },
          ],
        },
      ],
      tools: [
        {
          name: "TodoWrite",
          input_schema: { type: "object" },
        },
      ],
      tool_choice: { type: "any" },
    })
  })

  test("translates anthropic responses into responses results", () => {
    const response: AnthropicResponse = {
      id: "msg_123",
      type: "message",
      role: "assistant",
      model: "claude-opus-4.6",
      content: [
        {
          type: "thinking",
          thinking: "Thinking about it",
          signature: "opaque-signature",
        },
        {
          type: "tool_use",
          id: "call_1",
          name: "TodoWrite",
          input: { todos: [] },
        },
        {
          type: "text",
          text: "All set.",
        },
      ],
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: {
        input_tokens: 12,
        output_tokens: 4,
      },
    }

    const result = translateAnthropicResponseToResponsesResult(response, {
      model: "claude-opus-4.6",
      input: [],
    })

    expect(result.model).toBe("claude-opus-4.6")
    expect(result.status).toBe("completed")
    expect(result.output_text).toBe("All set.")
    expect(result.output.map((item) => item.type)).toEqual([
      "reasoning",
      "function_call",
      "message",
    ])
  })

  test("translates anthropic stream events into responses stream events", () => {
    const payload: ResponsesPayload = {
      model: "claude-opus-4.6",
      input: [],
      stream: true,
    }
    const state = createAnthropicResponsesStreamState(payload)

    const events = [
      {
        type: "message_start",
        message: {
          id: "msg_123",
          type: "message",
          role: "assistant",
          content: [],
          model: "claude-opus-4.6",
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: 5,
            output_tokens: 0,
          },
        },
      },
      {
        type: "content_block_start",
        index: 0,
        content_block: {
          type: "text",
          text: "",
        },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: {
          type: "text_delta",
          text: "hello",
        },
      },
      {
        type: "content_block_stop",
        index: 0,
      },
      {
        type: "message_delta",
        delta: {
          stop_reason: "end_turn",
          stop_sequence: null,
        },
        usage: {
          input_tokens: 5,
          output_tokens: 3,
        },
      },
      {
        type: "message_stop",
      },
    ].flatMap((event) =>
      translateAnthropicStreamEventToResponses(
        event as AnthropicStreamEventData,
        state,
      ),
    )

    events.push(...translateAnthropicResponsesStreamStateToResponses(state))

    expect(events.some((event) => event.type === "response.created")).toBe(true)
    expect(
      events.some((event) => event.type === "response.output_text.delta"),
    ).toBe(true)

    const completed = events.find(
      (event) => event.type === "response.completed",
    )
    expect(completed).toBeDefined()
    if (completed?.type === "response.completed") {
      expect(completed.response.output_text).toBe("hello")
      expect(completed.response.status).toBe("completed")
    }
  })
})
