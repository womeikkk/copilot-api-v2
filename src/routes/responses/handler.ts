import type { Context } from "hono"

import { streamSSE } from "hono/streaming"

import type { AnthropicStreamEventData } from "~/routes/messages/anthropic-types"

import { awaitApproval } from "~/lib/approval"
import {
  getConfig,
  isResponsesApiWebSearchEnabled,
  resolveMappedModel,
} from "~/lib/config"
import {
  createHandlerLogger,
  debugJson,
  debugJsonTail,
  logMappedModel,
  logRequestModel,
} from "~/lib/logger"
import { checkRateLimit } from "~/lib/rate-limit"
import { state } from "~/lib/state"
import { generateRequestIdFromPayload, getUUID } from "~/lib/utils"
import { prepareMessagesApiPayload } from "~/routes/messages/preprocess"
import { createMessages } from "~/services/copilot/create-messages"
import {
  createResponses,
  type ResponsesPayload,
  type ResponsesResult,
} from "~/services/copilot/create-responses"

import {
  createAnthropicResponsesStreamState,
  translateAnthropicResponseToResponsesResult,
  translateAnthropicResponsesStreamStateToResponses,
  translateAnthropicStreamEventToResponses,
  translateResponsesPayloadToAnthropicMessages,
} from "./messages-translation"
import { createStreamIdTracker, fixStreamIds } from "./stream-id-sync"
import {
  applyResponsesApiContextManagement,
  compactInputByLatestCompaction,
  getResponsesRequestOptions,
} from "./utils"

const logger = createHandlerLogger("responses-handler")

const RESPONSES_ENDPOINT = "/responses"

export const handleResponses = async (c: Context) => {
  await checkRateLimit(state)

  const payload = await c.req.json<ResponsesPayload>()
  logRequestModel("/v1/responses", payload.model)
  const requestedModel = payload.model
  payload.model = resolveMappedModel(payload.model)
  logMappedModel("/v1/responses", requestedModel, payload.model)
  debugJson(logger, "Responses request payload:", payload)

  // not support subagent marker for now , set sessionId = getUUID(requestId)
  const requestId = generateRequestIdFromPayload({ messages: payload.input })
  logger.debug("Generated request ID:", requestId)

  const sessionId = getUUID(requestId)
  logger.debug("Extracted session ID:", sessionId)

  useFunctionApplyPatch(payload)

  removeUnsupportedTools(payload)

  if (!isResponsesApiWebSearchEnabled()) {
    removeWebSearchTool(payload)
  }

  compactInputByLatestCompaction(payload)

  const selectedModel = state.models?.data.find(
    (model) => model.id === payload.model,
  )
  const supportsResponses =
    selectedModel?.supported_endpoints?.includes(RESPONSES_ENDPOINT) ?? false
  const supportsMessages =
    selectedModel?.supported_endpoints?.includes("/v1/messages") ?? false

  if (
    shouldRouteResponsesViaMessagesApi(
      payload.model,
      supportsResponses,
      supportsMessages,
    )
  ) {
    logger.debug(
      "Routing Responses request through the Messages API for mapped Claude model",
    )

    if (state.manualApprove) {
      await awaitApproval()
    }

    return await handleResponsesViaMessagesApi(c, payload, {
      requestId,
      sessionId,
      selectedModel,
    })
  }

  if (!supportsResponses) {
    return c.json(
      {
        error: {
          message:
            supportsMessages ?
              `Resolved model '${payload.model}' does not support the responses endpoint. It supports '/v1/messages', so either call '/v1/messages' or change modelMappings.`
            : "This model does not support the responses endpoint. Please choose a different model.",
          type: "invalid_request_error",
        },
      },
      400,
    )
  }

  if (state.manualApprove) {
    await awaitApproval()
  }

  return await handleNativeResponsesApi(c, payload, {
    requestId,
    sessionId,
    maxPromptTokens: selectedModel?.capabilities.limits.max_prompt_tokens,
  })
}

const handleResponsesViaMessagesApi = async (
  c: Context,
  payload: ResponsesPayload,
  options: {
    requestId: string
    sessionId: string
    selectedModel: NonNullable<typeof state.models>["data"][number] | undefined
  },
) => {
  const anthropicPayload = translateResponsesPayloadToAnthropicMessages(payload)

  prepareMessagesApiPayload(anthropicPayload, options.selectedModel)
  debugJson(
    logger,
    "Translated Responses payload to Messages payload:",
    anthropicPayload,
  )

  const response = await createMessages(anthropicPayload, undefined, {
    requestId: options.requestId,
    sessionId: options.sessionId,
  })

  if (isAsyncIterable(response)) {
    logger.debug("Forwarding Claude-backed Responses stream")
    return streamSSE(c, async (stream) => {
      const streamState = createAnthropicResponsesStreamState(payload)

      for await (const event of response) {
        const data = event.data ?? ""
        if (!data || data === "[DONE]") {
          continue
        }

        let parsed: unknown
        try {
          parsed = JSON.parse(data)
        } catch {
          continue
        }

        const translatedEvents = translateAnthropicStreamEventToResponses(
          parsed as AnthropicStreamEventData,
          streamState,
        )

        for (const translatedEvent of translatedEvents) {
          await stream.writeSSE({
            event: translatedEvent.type,
            data: JSON.stringify(translatedEvent),
          })
        }
      }

      for (const translatedEvent of translateAnthropicResponsesStreamStateToResponses(
        streamState,
      )) {
        await stream.writeSSE({
          event: translatedEvent.type,
          data: JSON.stringify(translatedEvent),
        })
      }
    })
  }

  debugJsonTail(logger, "Claude-backed Messages result:", {
    value: response,
    tailLength: 400,
  })

  return c.json(translateAnthropicResponseToResponsesResult(response, payload))
}

const handleNativeResponsesApi = async (
  c: Context,
  payload: ResponsesPayload,
  options: {
    requestId: string
    sessionId: string
    maxPromptTokens?: number
  },
) => {
  applyResponsesApiContextManagement(payload, options.maxPromptTokens)

  debugJson(logger, "Translated Responses payload:", payload)

  const { vision, initiator } = getResponsesRequestOptions(payload)
  const response = await createResponses(payload, {
    vision,
    initiator,
    requestId: options.requestId,
    sessionId: options.sessionId,
  })

  if (isStreamingRequested(payload) && isAsyncIterable(response)) {
    logger.debug("Forwarding native Responses stream")
    return streamSSE(c, async (stream) => {
      const idTracker = createStreamIdTracker()

      for await (const chunk of response) {
        debugJson(logger, "Responses stream chunk:", chunk)

        const processedData = fixStreamIds(
          (chunk as { data?: string }).data ?? "",
          (chunk as { event?: string }).event,
          idTracker,
        )

        await stream.writeSSE({
          id: (chunk as { id?: string }).id,
          event: (chunk as { event?: string }).event,
          data: processedData,
        })
      }
    })
  }

  debugJsonTail(logger, "Forwarding native Responses result:", {
    value: response,
    tailLength: 400,
  })
  return c.json(response as ResponsesResult)
}

const shouldRouteResponsesViaMessagesApi = (
  model: string,
  supportsResponses: boolean,
  supportsMessages: boolean,
): boolean => {
  return (
    !supportsResponses
    && supportsMessages
    && model.toLowerCase().includes("claude")
  )
}

const isAsyncIterable = <T>(value: unknown): value is AsyncIterable<T> =>
  Boolean(value)
  && typeof (value as AsyncIterable<T>)[Symbol.asyncIterator] === "function"

const isStreamingRequested = (payload: ResponsesPayload): boolean =>
  Boolean(payload.stream)

const useFunctionApplyPatch = (payload: ResponsesPayload): void => {
  const config = getConfig()
  const useFunctionApplyPatch = config.useFunctionApplyPatch ?? true
  if (useFunctionApplyPatch) {
    logger.debug("Using function tool apply_patch for responses")
    if (Array.isArray(payload.tools)) {
      const toolsArr = payload.tools
      for (let i = 0; i < toolsArr.length; i++) {
        const t = toolsArr[i]
        if (t.type === "custom" && t.name === "apply_patch") {
          toolsArr[i] = {
            type: "function",
            name: t.name,
            description: "Use the `apply_patch` tool to edit files",
            parameters: {
              type: "object",
              properties: {
                input: {
                  type: "string",
                  description: "The entire contents of the apply_patch command",
                },
              },
              required: ["input"],
            },
            strict: false,
          }
        }
      }
    }
  }
}

const removeWebSearchTool = (payload: ResponsesPayload): void => {
  if (!Array.isArray(payload.tools) || payload.tools.length === 0) return

  payload.tools = payload.tools.filter((t) => {
    return t.type !== "web_search"
  })
}

const COPILOT_UNSUPPORTED_TOOL_TYPES = new Set(["image_generation"])

export const removeUnsupportedTools = (payload: ResponsesPayload): void => {
  if (!Array.isArray(payload.tools) || payload.tools.length === 0) return

  const dropped: Array<string> = []
  payload.tools = payload.tools.filter((t) => {
    const type = t.type as string
    if (COPILOT_UNSUPPORTED_TOOL_TYPES.has(type)) {
      dropped.push(type)
      return false
    }
    return true
  })
  if (dropped.length > 0) {
    logger.debug("Removed unsupported tools:", dropped)
  }
}
