import { uid } from "./utils";

export async function fetchBootstrap() {
  return requestJson("/api/bootstrap", {}, "加载运行状态失败");
}

export async function fetchCommands() {
  return requestJson("/api/commands", {}, "加载命令失败");
}

export async function fetchConversations() {
  return requestJson("/api/conversations", {}, "加载对话失败");
}

export async function fetchRuntimeLog() {
  return requestJson("/api/logs/runtime", {}, "加载运行日志失败");
}

export async function createConversation(title = "新对话") {
  return requestJson("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  }, "创建对话失败");
}

export async function deleteConversation(conversationId) {
  return requestJson(`/api/conversations/${encodeURIComponent(conversationId)}`, {
    method: "DELETE",
  }, "删除对话失败");
}

export async function sendChat(conversationId, message, requestId, reasoningEnabled, controller, onEvent) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation_id: conversationId,
      message,
      request_id: requestId,
      reasoning_enabled: reasoningEnabled,
    }),
    signal: controller.signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`请求失败：${response.status}`);
  }
  await readEventStream(response.body, onEvent);
}

export async function cancelGeneration(requestId) {
  await requestJson("/api/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request_id: requestId }),
  }, "取消生成失败");
}

async function requestJson(url, options = {}, label = "请求失败") {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`${label}：${response.status}`);
  }
  return response.json();
}

async function readEventStream(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";
    for (const chunk of chunks) {
      const line = chunk.split("\n").find((item) => item.startsWith("data: "));
      if (line) onEvent(JSON.parse(line.slice(6)));
    }
  }
}

export function applyEvent(event, setMessages, setStatus) {
  if (event.type === "done") {
    setMessages((items) => finalizeCurrentTurnAssistants(items));
    return;
  }
  if (event.type === "status") {
    setStatus((current) => ({ ...current, ...event }));
    return;
  }
  if (event.type === "clear") {
    setMessages([]);
    return;
  }
  if (event.type === "user_message") {
    setMessages((items) => [...items, { id: uid(), role: "user", text: event.text }]);
    return;
  }
  if (event.type === "assistant_start") {
    setMessages((items) => {
      const last = items[items.length - 1];
      if (last?.role === "assistant" && !last.streamComplete) return items;
      return [...items, createAssistantMessage()];
    });
    return;
  }
  if (event.type === "assistant_delta") {
    setMessages((items) =>
      updateCurrentAssistant(items, (m) => ({
        ...completeLatestReasoningStep(m),
        text: m.text + event.text,
        streamComplete: false,
      })),
    );
    return;
  }
  if (event.type === "reasoning_delta") {
    setMessages((items) =>
      updateCurrentAssistant(items, (m) => ({
        ...appendReasoningStep(m, event.text),
        reasoningComplete: false,
        streamComplete: false,
      })),
    );
    return;
  }
  if (event.type === "tool_call_start") {
    setMessages((items) =>
      updateCurrentAssistant(items, (m) => {
        const completed = completeLatestReasoningStep(m);
        return {
          ...completed,
          reasoningOpen: false,
          streamComplete: false,
          steps: upsertToolStep(completed.steps, {
            id: event.id,
            type: "tool",
            name: event.name,
            arguments: event.arguments,
            argumentsSummary: event.argumentsSummary,
            status: event.status ?? "running",
          }),
          tools: upsertTool(m.tools, {
            id: event.id,
            name: event.name,
            arguments: event.arguments,
            argumentsSummary: event.argumentsSummary,
            status: event.status ?? "running",
          }),
        };
      }),
    );
    return;
  }
  if (event.type === "tool_call_result") {
    setMessages((items) =>
      updateCurrentAssistant(items, (m) => ({
        ...m,
        streamComplete: false,
        steps: upsertToolStep(m.steps, {
          id: event.id,
          type: "tool",
          name: event.name,
          arguments: event.arguments,
          argumentsSummary: event.argumentsSummary,
          status: event.status ?? "completed",
          result: event.result,
          summary: event.summary,
        }),
        tools: upsertTool(m.tools, {
          id: event.id,
          name: event.name,
          arguments: event.arguments,
          argumentsSummary: event.argumentsSummary,
          status: event.status ?? "completed",
          result: event.result,
          summary: event.summary,
        }),
      })),
    );
    return;
  }
  if (event.type === "assistant_notice") {
    setMessages((items) =>
      updateCurrentAssistant(items, (m) =>
        finalizeAssistantMessage({
          ...m,
          text: m.text ? `${m.text}\n${event.text}` : event.text,
        }),
      ),
    );
    return;
  }
  if (event.type === "notice") {
    if (event.text === "生成已停止") {
      setMessages((items) =>
        updateCurrentAssistant(items, (m) =>
          finalizeAssistantMessage({
            ...m,
            text: m.text ? `${m.text}\n${event.text}` : event.text,
          }),
        ),
      );
      return;
    }
    appendNotice(setMessages, event.tone, event.text);
  }
}

function appendNotice(setMessages, tone, text) {
  setMessages((items) => [...items, { id: uid(), role: "notice", tone, text }]);
}

function updateCurrentAssistant(items, updater) {
  const next = [...items];
  const index = next.length - 1;
  if (next[index]?.role !== "assistant") {
    next.push(updater(createAssistantMessage()));
    return next;
  }
  next[index] = updater(next[index]);
  return next;
}

function finalizeCurrentTurnAssistants(items) {
  const next = [...items];
  let changed = false;
  for (let index = next.length - 1; index >= 0; index -= 1) {
    const item = next[index];
    if (item?.role === "user") break;
    if (item?.role !== "assistant") continue;
    next[index] = finalizeAssistantMessage(item);
    changed = true;
  }
  return changed ? next : items;
}

function finalizeAssistantMessage(message) {
  return {
    ...message,
    reasoningOpen: false,
    reasoningComplete: Boolean(message.reasoning || message.steps?.some((step) => step.type === "reasoning")),
    streamComplete: true,
    completedKey: uid(),
    steps: (message.steps ?? []).map((step) =>
      step.type === "reasoning" ? { ...step, complete: true, open: false } : step,
    ),
  };
}

function createAssistantMessage() {
  return {
    id: uid(),
    role: "assistant",
    text: "",
    reasoning: "",
    reasoningOpen: true,
    reasoningComplete: false,
    streamComplete: false,
    completedKey: "",
    steps: [],
    tools: [],
  };
}

function appendReasoningStep(message, text) {
  const steps = message.steps ?? [];
  const last = steps[steps.length - 1];
  let nextSteps;
  if (last?.type === "reasoning" && !last.complete) {
    nextSteps = [
      ...steps.slice(0, -1),
      {
        ...last,
        text: `${last.text ?? ""}${text}`,
        open: true,
      },
    ];
  } else {
    nextSteps = [
      ...steps,
      {
        id: uid(),
        type: "reasoning",
        text,
        open: true,
        complete: false,
      },
    ];
  }
  return {
    ...message,
    reasoning: (message.reasoning ?? "") + text,
    reasoningOpen: true,
    steps: nextSteps,
  };
}

function completeLatestReasoningStep(message) {
  const steps = message.steps ?? [];
  const last = steps[steps.length - 1];
  if (last?.type !== "reasoning" || last.complete) {
    return message;
  }
  return {
    ...message,
    reasoningOpen: false,
    steps: [
      ...steps.slice(0, -1),
      {
        ...last,
        complete: true,
        open: false,
      },
    ],
  };
}

function upsertTool(items = [], nextTool) {
  const normalized = {
    ...nextTool,
    id: nextTool.id || `${nextTool.name}-${items.length}`,
  };
  const index = items.findIndex((item) => item.id === normalized.id);
  if (index === -1) return [...items, normalized];
  const next = [...items];
  next[index] = { ...next[index], ...normalized };
  return next;
}

function upsertToolStep(items = [], nextTool) {
  const normalized = {
    ...nextTool,
    id: nextTool.id || `${nextTool.name}-${items.length}`,
    type: "tool",
  };
  const index = items.findIndex((item) => item.type === "tool" && item.id === normalized.id);
  if (index === -1) return [...items, normalized];
  const next = [...items];
  next[index] = { ...next[index], ...normalized };
  return next;
}
