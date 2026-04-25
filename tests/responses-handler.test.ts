import { beforeEach, describe, expect, mock, test } from "bun:test"
import { Hono } from "hono"

import type { AnthropicResponse } from "~/routes/messages/anthropic-types"
import type { ResponsesPayload } from "~/services/copilot/create-responses"

const actualConfigModule = await import("../src/lib/config")
const actualRateLimitModule = await import("../src/lib/rate-limit")
const actualStateModule = await import("../src/lib/state")
const actualUtilsModule = await import("../src/lib/utils")

const state = {
  ...actualStateModule.state,
  manualApprove: false,
  verbose: false,
}

let resolveMappedModelImpl = (model: string) => model

const createMessages = mock(
  (): Promise<AnthropicResponse> =>
    Promise.resolve({
      id: "msg_123",
      type: "message",
      role: "assistant",
      model: "claude-opus-4.6",
      content: [{ type: "text", text: "hello from claude" }],
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: {
        input_tokens: 12,
        output_tokens: 4,
      },
    }),
)

const createResponses = mock(() =>
  Promise.resolve({
    id: "resp_123",
    object: "response" as const,
    created_at: 0,
    model: "gpt-5.4",
    output: [],
    output_text: "native responses",
    status: "completed",
    usage: null,
    error: null,
    incomplete_details: null,
    instructions: null,
    metadata: null,
    parallel_tool_calls: true,
    temperature: null,
    tool_choice: "auto",
    tools: [],
    top_p: null,
  }),
)

await mock.module("~/lib/state", () => ({
  ...actualStateModule,
  state,
}))
await mock.module("~/lib/rate-limit", () => ({
  ...actualRateLimitModule,
  checkRateLimit: async () => {},
}))
await mock.module("~/lib/config", () => ({
  ...actualConfigModule,
  getConfig: () => ({}),
  isResponsesApiWebSearchEnabled: () => true,
  resolveMappedModel: (model: string) => resolveMappedModelImpl(model),
}))
await mock.module("~/lib/utils", () => ({
  ...actualUtilsModule,
}))
await mock.module("~/services/copilot/create-messages", () => ({
  createMessages,
}))
await mock.module("~/services/copilot/create-responses", () => ({
  createResponses,
}))

const { responsesRoutes } = await import("../src/routes/responses/route")

const createApp = () => {
  const app = new Hono()
  app.route("/v1/responses", responsesRoutes)
  return app
}

const createModels = () => ({
  object: "list" as const,
  data: [
    {
      capabilities: {
        family: "claude",
        limits: {},
        object: "model_capabilities" as const,
        supports: {},
        tokenizer: "o200k_base",
        type: "chat" as const,
      },
      id: "claude-opus-4.6",
      model_picker_enabled: true,
      name: "claude-opus-4.6",
      object: "model" as const,
      preview: false,
      vendor: "anthropic",
      version: "1",
      supported_endpoints: ["/v1/messages"],
    },
  ],
})

beforeEach(() => {
  state.manualApprove = false
  state.verbose = false
  state.models = createModels()
  resolveMappedModelImpl = (model) =>
    model === "gpt-5.5" ? "claude-opus-4.6" : model
  createMessages.mockClear()
  createResponses.mockClear()
})

describe("responses handler", () => {
  test("routes mapped claude models through the messages api", async () => {
    const app = createApp()

    const payload: ResponsesPayload = {
      model: "gpt-5.5",
      input: [
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "hello" }],
        },
      ],
    }

    const response = await app.request("/v1/responses", {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify(payload),
    })

    expect(response.status).toBe(200)
    expect(createMessages).toHaveBeenCalledTimes(1)
    expect(createResponses).not.toHaveBeenCalled()

    const json = await response.json()
    expect(json.model).toBe("claude-opus-4.6")
    expect(json.output_text).toBe("hello from claude")
  })
})
