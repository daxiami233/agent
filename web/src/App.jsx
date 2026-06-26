import { useEffect, useMemo, useRef, useState } from "react";
import { uid } from "./utils";
import {
  fetchBootstrap,
  fetchCommands,
  fetchConversations,
  fetchRuntimeLog,
  createConversation as createRemoteConversation,
  deleteConversation as deleteRemoteConversation,
  sendChat,
  cancelGeneration,
  applyEvent,
} from "./api";
import Message from "./components/Message";
import RuntimeStatus from "./components/RuntimeStatus";
import CommandMenu from "./components/CommandMenu";

const LEGACY_STORAGE_KEY = "agent-runtime-web-state-v1";
const ACTIVE_CONVERSATION_KEY = "agent-runtime-active-conversation-v1";
const REASONING_ENABLED_KEY = "agent-runtime-reasoning-enabled-v1";
const BOTTOM_FOLLOW_THRESHOLD_PX = 120;

function isNearBottom(element) {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= BOTTOM_FOLLOW_THRESHOLD_PX;
}

function createLocalConversation(title = "新对话") {
  return {
    id: uid(),
    title,
    messages: [],
    createdAt: Date.now(),
    context: "输入上下文：未配置",
    contextUsed: 0,
    inputBudgetTokens: null,
    contextWindow: null,
  };
}

function normalizeConversation(conversation) {
  return {
    id: conversation.id || uid(),
    title: conversation.title || "新对话",
    messages: Array.isArray(conversation.messages)
      ? conversation.messages.map((message) => ({
          id: message.id || uid(),
          role: message.role,
          text: message.text ?? message.content ?? "",
          reasoning: message.reasoning,
          reasoningOpen: Boolean(message.reasoningOpen),
          reasoningComplete: Boolean(message.reasoningComplete),
          streamComplete: Boolean(message.streamComplete),
          completedKey: message.completedKey ?? "",
          steps: Array.isArray(message.steps) ? message.steps : [],
          tools: Array.isArray(message.tools) ? message.tools : [],
          tone: message.tone,
        }))
      : [],
    createdAt: conversation.createdAt || Date.now(),
    updatedAt: conversation.updatedAt || conversation.createdAt || Date.now(),
    context: conversation.context ?? "输入上下文：未配置",
    contextUsed: conversation.contextUsed ?? 0,
    inputBudgetTokens: conversation.inputBudgetTokens ?? null,
    contextWindow: conversation.contextWindow ?? null,
  };
}

function promptsFromConversations(items) {
  const prompts = [];
  for (const conversation of items) {
    for (const message of conversation.messages) {
      if (message.role === "user" && message.text && !prompts.includes(message.text)) {
        prompts.push(message.text);
      }
    }
  }
  return prompts.slice(0, 80);
}

function promptTitle(value) {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (!normalized) return "新对话";
  return normalized.length > 34 ? `${normalized.slice(0, 31)}...` : normalized;
}

function commandTitle(value) {
  const name = value.trim().split(/\s+/, 1)[0];
  if (name === "/help") return "命令";
  if (name === "/model") return "模型";
  if (name === "/status") return "状态";
  return "命令";
}

function applyCommandEvent(event, setCommandPanel, setStatus) {
  if (event.type === "done") return;
  if (event.type === "status") {
    setStatus((current) => ({ ...current, ...event }));
    return;
  }
  if (event.type !== "notice") return;
  setCommandPanel((current) => {
    const panel = current ?? { id: uid(), title: "命令", command: "", lines: [], tone: "info" };
    return {
      ...panel,
      tone: event.tone === "error" ? "error" : panel.tone,
      lines: [...panel.lines, event.text],
    };
  });
}

function CommandPanel({ panel, onClose }) {
  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <section className={`commandPanel ${panel.tone ?? "info"}`}>
      <header>
        <h2>{panel.title}</h2>
        <div className="commandPanelActions">
          {panel.command && <code>{panel.command}</code>}
          <button type="button" onClick={onClose}>关闭</button>
        </div>
      </header>
      <div className="commandPanelBody">
        {panel.lines.length === 0 ? (
          <p>执行中...</p>
        ) : (
          panel.lines.map((line, index) => <pre key={`${panel.id}-${index}`}>{line}</pre>)
        )}
      </div>
    </section>
  );
}

function RuntimeLogModal({ log, loading, error, onRefresh, onClose }) {
  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <div className="modalBackdrop logModalBackdrop" role="presentation">
      <section
        className="runtimeLogDialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="runtime-log-title"
      >
        <header>
          <div>
            <h2 id="runtime-log-title">运行日志</h2>
            <p title={log.path}>{log.path || "未找到日志路径"}</p>
          </div>
          <div className="runtimeLogActions">
            <button type="button" onClick={onRefresh} disabled={loading}>
              {loading ? "刷新中" : "刷新"}
            </button>
            <button type="button" onClick={onClose}>关闭</button>
          </div>
        </header>
        {error && <div className="runtimeLogError">{error}</div>}
        <pre className="runtimeLogContent">
          {loading && !log.content
            ? "加载中..."
            : log.content || (log.exists ? "日志为空。" : "日志文件还不存在。")}
        </pre>
      </section>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="typing" role="status" aria-live="polite">
      <span className="typingText">思考中</span>
      <span className="typingDots" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
    </div>
  );
}

export default function App() {
  const [conversationState, setConversationState] = useState(() => {
    const first = createLocalConversation();
    return { items: [first], activeId: first.id };
  });
  const [input, setInput] = useState("");
  const [promptHistory, setPromptHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(null);
  const [status, setStatus] = useState({
    cwd: "agent",
    cwdPath: "",
    model: "未配置",
    context: "输入上下文：未配置",
    contextUsed: 0,
    inputBudgetTokens: null,
    contextWindow: null,
    apiMode: "auto",
  });
  const [commands, setCommands] = useState([]);
  const [reasoningEnabled, setReasoningEnabled] = useState(() => {
    try {
      return localStorage.getItem(REASONING_ENABLED_KEY) !== "false";
    } catch {
      return true;
    }
  });
  const [streaming, setStreaming] = useState(false);
  const [streamingConversationId, setStreamingConversationId] = useState("");
  const [error, setError] = useState("");
  const [commandPanel, setCommandPanel] = useState(null);
  const [activeCommandIndex, setActiveCommandIndex] = useState(0);
  const [commandMenuDismissed, setCommandMenuDismissed] = useState(false);
  const [conversationsLoaded, setConversationsLoaded] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [showJumpToBottom, setShowJumpToBottom] = useState(false);
  const [logModalOpen, setLogModalOpen] = useState(false);
  const [runtimeLog, setRuntimeLog] = useState({
    path: "",
    content: "",
    exists: false,
  });
  const [runtimeLogLoading, setRuntimeLogLoading] = useState(false);
  const [runtimeLogError, setRuntimeLogError] = useState("");
  const transcriptRef = useRef(null);
  const textareaRef = useRef(null);
  const controllerRef = useRef(null);
  const requestIdRef = useRef("");
  const autoFollowRef = useRef(true);

  const conversations = conversationState.items;
  const activeConversation =
    conversations.find((conversation) => conversation.id === conversationState.activeId) ?? conversations[0];
  const activeMessages = activeConversation?.messages ?? [];

  useEffect(() => {
    try {
      localStorage.removeItem(LEGACY_STORAGE_KEY);
    } catch {
      // localStorage may be unavailable in restricted browser contexts.
    }
  }, []);

  useEffect(() => {
    if (!conversationsLoaded) return;
    try {
      localStorage.setItem(ACTIVE_CONVERSATION_KEY, conversationState.activeId);
    } catch {
      // localStorage may be unavailable in restricted browser contexts.
    }
  }, [conversationState.activeId, conversationsLoaded]);

  useEffect(() => {
    try {
      localStorage.setItem(REASONING_ENABLED_KEY, String(reasoningEnabled));
    } catch {
      // localStorage may be unavailable in restricted browser contexts.
    }
  }, [reasoningEnabled]);

  function updateConversationMessages(conversationId, updater) {
    setConversationState((current) => ({
      ...current,
      items: current.items.map((conversation) => {
        if (conversation.id !== conversationId) return conversation;
        return {
          ...conversation,
          messages: typeof updater === "function" ? updater(conversation.messages) : updater,
        };
      }),
    }));
  }

  function renameConversationFromPrompt(conversationId, prompt) {
    setConversationState((current) => ({
      ...current,
      items: current.items.map((conversation) => {
        if (conversation.id !== conversationId || !["New chat", "新对话"].includes(conversation.title)) {
          return conversation;
        }
        return { ...conversation, title: promptTitle(prompt) };
      }),
    }));
  }

  function updateConversationStatus(conversationId, event) {
    setConversationState((current) => ({
      ...current,
      items: current.items.map((conversation) => {
        if (conversation.id !== conversationId) return conversation;
        return {
          ...conversation,
          context: event.context ?? conversation.context,
          contextUsed: event.contextUsed ?? conversation.contextUsed,
          inputBudgetTokens: event.inputBudgetTokens ?? conversation.inputBudgetTokens,
          contextWindow: event.contextWindow ?? conversation.contextWindow,
        };
      }),
    }));
  }

  function applyConversationStatus(conversation) {
    if (!conversation) return;
    setStatus((current) => ({
      ...current,
      context: conversation.context ?? current.context,
      contextUsed: conversation.contextUsed ?? current.contextUsed,
      inputBudgetTokens: conversation.inputBudgetTokens ?? current.inputBudgetTokens,
      contextWindow: conversation.contextWindow ?? current.contextWindow,
    }));
  }

  useEffect(() => {
    let mounted = true;
    Promise.all([fetchBootstrap(), fetchCommands(), fetchConversations()])
      .then(async ([boot, commandData, conversationData]) => {
        let items = (conversationData.items ?? []).map(normalizeConversation);
        if (items.length === 0) {
          items = [normalizeConversation(await createRemoteConversation())];
        }
        let savedActiveId = "";
        try {
          savedActiveId = localStorage.getItem(ACTIVE_CONVERSATION_KEY) ?? "";
        } catch {
          savedActiveId = "";
        }
        const activeId = items.some((conversation) => conversation.id === savedActiveId)
          ? savedActiveId
          : items[0].id;
        if (!mounted) return;
        setCommands(commandData.commands ?? []);
        setPromptHistory(promptsFromConversations(items));
        setConversationState({ items, activeId });
        setConversationsLoaded(true);
        for (const event of boot.events ?? []) {
          applyEvent(event, (updater) => updateConversationMessages(activeId, updater), setStatus);
        }
        applyConversationStatus(items.find((conversation) => conversation.id === activeId) ?? items[0]);
      })
      .catch((err) => setError(String(err)));
    return () => {
      mounted = false;
    };
  }, []);

  function scrollTranscriptToBottom(behavior = "auto") {
    const el = transcriptRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior });
    autoFollowRef.current = true;
    setShowJumpToBottom(false);
  }

  function updateScrollFollowState() {
    const el = transcriptRef.current;
    if (!el) return;
    const atBottom = isNearBottom(el);
    autoFollowRef.current = atBottom;
    setShowJumpToBottom(!atBottom);
  }

  function jumpToBottom() {
    scrollTranscriptToBottom("smooth");
  }

  useEffect(() => {
    autoFollowRef.current = true;
    setShowJumpToBottom(false);
    requestAnimationFrame(() => scrollTranscriptToBottom());
  }, [conversationState.activeId]);

  useEffect(() => {
    if (!autoFollowRef.current) return;
    requestAnimationFrame(() => scrollTranscriptToBottom());
  }, [activeMessages, streaming]);

  useEffect(() => {
    if (!conversationsLoaded) return;
    applyConversationStatus(activeConversation);
  }, [
    activeConversation?.id,
    activeConversation?.context,
    activeConversation?.contextUsed,
    activeConversation?.inputBudgetTokens,
    activeConversation?.contextWindow,
    conversationsLoaded,
  ]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    const nextHeight = Math.min(textarea.scrollHeight, 192);
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > 192 ? "auto" : "hidden";
  }, [input]);

  const matches = useMemo(() => {
    const value = input.trim();
    if (commandMenuDismissed) return [];
    if (!value.startsWith("/") && !value.startsWith(":")) return [];
    if (/\s/.test(value)) return [];
    return commands.filter((command) => command.name.startsWith(value)).slice(0, 7);
  }, [commandMenuDismissed, commands, input]);

  useEffect(() => {
    setActiveCommandIndex(0);
  }, [matches.length, input]);

  function rememberPrompt(value) {
    if (value.startsWith("/") || value.startsWith(":")) return;
    setPromptHistory((items) => {
      const next = [value, ...items.filter((item) => item !== value)];
      return next.slice(0, 80);
    });
    setHistoryIndex(null);
  }

  async function submit() {
    await submitText(input.trim(), activeConversation.id);
  }

  async function submitText(value, conversationId) {
    if (!value || streaming) return;
    autoFollowRef.current = true;
    setShowJumpToBottom(false);
    requestAnimationFrame(() => scrollTranscriptToBottom());
    const requestId = uid();
    const controller = new AbortController();
    requestIdRef.current = requestId;
    controllerRef.current = controller;
    setStreamingConversationId(conversationId);
    setInput("");
    setError("");
    setCommandMenuDismissed(false);
    setStreaming(true);
    const isCommand = value.startsWith("/") || value.startsWith(":");
    if (isCommand) {
      setCommandPanel({
        id: uid(),
        title: commandTitle(value),
        command: value,
        lines: [],
        tone: "info",
      });
    } else {
      rememberPrompt(value);
      renameConversationFromPrompt(conversationId, value);
    }
    try {
      await sendChat(conversationId, value, requestId, reasoningEnabled, controller, (event) => {
        if (isCommand) {
          applyCommandEvent(event, setCommandPanel, setStatus);
        } else {
          if (!reasoningEnabled && event.type === "reasoning_delta") return;
          if (event.type === "status") updateConversationStatus(conversationId, event);
          applyEvent(event, (updater) => updateConversationMessages(conversationId, updater), setStatus);
        }
      });
    } catch (err) {
      if (err.name !== "AbortError") setError(String(err));
    } finally {
      controllerRef.current = null;
      requestIdRef.current = "";
      setStreaming(false);
      setStreamingConversationId("");
    }
  }

  async function stopGeneration() {
    const requestId = requestIdRef.current;
    const conversationId = streamingConversationId || activeConversation.id;
    if (!streaming || !requestId) return;
    applyEvent(
      { type: "assistant_notice", text: "生成已停止" },
      (updater) => updateConversationMessages(conversationId, updater),
      setStatus,
    );
    try {
      await cancelGeneration(requestId);
    } catch {
      // 即使取消请求失败，本地中断也会立即释放界面。
    } finally {
      controllerRef.current?.abort();
    }
  }

  function onKeyDown(event) {
    if (event.nativeEvent?.isComposing || event.isComposing || event.keyCode === 229) {
      return;
    }
    if (matches.length > 0 && event.key === "ArrowDown") {
      event.preventDefault();
      setActiveCommandIndex((index) => (index + 1) % matches.length);
      return;
    }
    if (matches.length > 0 && event.key === "ArrowUp") {
      event.preventDefault();
      setActiveCommandIndex((index) => (index - 1 + matches.length) % matches.length);
      return;
    }
    if (matches.length > 0 && (event.key === "Enter" || event.key === "Tab")) {
      event.preventDefault();
      executeCommand(matches[activeCommandIndex].name);
      return;
    }
    if (matches.length === 0 && event.key === "ArrowUp" && promptHistory.length > 0) {
      event.preventDefault();
      const nextIndex = historyIndex === null ? 0 : Math.min(historyIndex + 1, promptHistory.length - 1);
      setHistoryIndex(nextIndex);
      setInput(promptHistory[nextIndex]);
      setCommandMenuDismissed(true);
      return;
    }
    if (matches.length === 0 && event.key === "ArrowDown" && historyIndex !== null) {
      event.preventDefault();
      const nextIndex = historyIndex - 1;
      if (nextIndex < 0) {
        setHistoryIndex(null);
        setInput("");
      } else {
        setHistoryIndex(nextIndex);
        setInput(promptHistory[nextIndex]);
      }
      setCommandMenuDismissed(true);
      return;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  function executeCommand(name) {
    submitText(name, activeConversation.id);
  }

  async function createNewConversation() {
    try {
      const conversation = normalizeConversation(await createRemoteConversation());
      setConversationState((current) => ({
        items: [conversation, ...current.items],
        activeId: conversation.id,
      }));
      setInput("");
      setError("");
      setCommandMenuDismissed(false);
      requestAnimationFrame(() => document.querySelector(".composer textarea")?.focus());
    } catch (err) {
      setError(String(err));
    }
  }

  async function deleteConversation(conversationId) {
    try {
      await deleteRemoteConversation(conversationId);
    } catch (err) {
      setError(String(err));
      return;
    }
    setDeleteTarget(null);
    let replacement = null;
    if (conversations.length === 1) {
      try {
        replacement = normalizeConversation(await createRemoteConversation());
      } catch (err) {
        setError(String(err));
        replacement = createLocalConversation();
      }
    }
    setConversationState((current) => {
      const remaining = current.items.filter((conversation) => conversation.id !== conversationId);
      if (remaining.length === 0) {
        const next = replacement ?? createLocalConversation();
        return { items: [next], activeId: next.id };
      }
      const activeId = current.activeId === conversationId ? remaining[0].id : current.activeId;
      return { items: remaining, activeId };
    });
  }

  async function openRuntimeLog() {
    setLogModalOpen(true);
    await refreshRuntimeLog();
  }

  async function refreshRuntimeLog() {
    setRuntimeLogLoading(true);
    setRuntimeLogError("");
    try {
      const payload = await fetchRuntimeLog();
      setRuntimeLog({
        path: payload.path || "",
        content: payload.content || "",
        exists: Boolean(payload.exists),
      });
      if (payload.error) setRuntimeLogError(payload.error);
    } catch (err) {
      setRuntimeLogError(String(err));
    } finally {
      setRuntimeLogLoading(false);
    }
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="sideHeader">
          <div>
            <div className="brand">智能体运行时</div>
            <div className="subtitle">本地网页智能体</div>
          </div>
          <button type="button" onClick={createNewConversation}>新对话</button>
        </div>
        <nav className="conversationList" aria-label="对话列表">
          {conversations.map((conversation) => (
            <div
              key={conversation.id}
              className={`conversationItem ${conversation.id === activeConversation.id ? "active" : ""}`}
            >
              <button
                type="button"
                className="conversationOpen"
                onClick={() => setConversationState((current) => ({ ...current, activeId: conversation.id }))}
              >
                <span>{conversation.title}</span>
              </button>
              <button
                type="button"
                className="conversationDelete"
                aria-label={`删除 ${conversation.title}`}
                onClick={() => setDeleteTarget(conversation)}
              >
                ×
              </button>
            </div>
          ))}
        </nav>
      </aside>

      <section className="chatPane">
        <section
          ref={transcriptRef}
          className="transcript"
          onScroll={updateScrollFollowState}
        >
          {activeMessages.length === 0 ? (
            <div className="welcome">
              <h1>有什么可以帮你？</h1>
              <p>输入问题，或使用斜杠命令查看当前运行状态。</p>
            </div>
          ) : (
            activeMessages.map((message) => <Message key={message.id} message={message} />)
          )}
          {streaming && streamingConversationId === activeConversation.id && (
            <TypingIndicator />
          )}
        </section>
        {showJumpToBottom && (
          <button
            type="button"
            className="jumpToBottom"
            onClick={jumpToBottom}
            aria-label="跳到底部"
            title="跳到底部"
          >
            ↓
          </button>
        )}

        <section className="composerWrap">
          <div className="composerStack">
            <div className="commandOverlay">
              <CommandMenu
                matches={matches}
                activeIndex={activeCommandIndex}
                onChoose={executeCommand}
                onHover={setActiveCommandIndex}
              />
            </div>
            {commandPanel && (
              <CommandPanel panel={commandPanel} onClose={() => setCommandPanel(null)} />
            )}
            {error && <div className="error">{error}</div>}
            <div className="composer">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(event) => {
                  setInput(event.target.value);
                  setCommandMenuDismissed(false);
                }}
                onKeyDown={onKeyDown}
                placeholder="输入消息"
                rows={1}
              />
              <button
                type="button"
                className={`reasonToggle ${reasoningEnabled ? "active" : ""}`}
                aria-pressed={reasoningEnabled}
                title={reasoningEnabled ? "关闭思考" : "打开思考"}
                onClick={() => setReasoningEnabled((enabled) => !enabled)}
              >
                {reasoningEnabled ? "思考开" : "思考关"}
              </button>
              <button
                type="button"
                className={streaming ? "stop" : ""}
                onClick={streaming ? stopGeneration : submit}
                disabled={!streaming && !input.trim()}
              >
                {streaming ? "停止" : "发送"}
              </button>
            </div>
            <RuntimeStatus status={status} />
          </div>
        </section>
      </section>
      <button
        type="button"
        className="runtimeLogButton"
        onClick={openRuntimeLog}
        title="查看运行日志"
        aria-label="查看运行日志"
      >
        日志
      </button>
      {logModalOpen && (
        <RuntimeLogModal
          log={runtimeLog}
          loading={runtimeLogLoading}
          error={runtimeLogError}
          onRefresh={refreshRuntimeLog}
          onClose={() => setLogModalOpen(false)}
        />
      )}
      {deleteTarget && (
        <div className="modalBackdrop" role="presentation">
          <section
            className="confirmDialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-dialog-title"
          >
            <h2 id="delete-dialog-title">删除对话？</h2>
            <p>{deleteTarget.title}</p>
            <div className="confirmActions">
              <button type="button" onClick={() => setDeleteTarget(null)}>
                取消
              </button>
              <button
                type="button"
                className="danger"
                onClick={() => deleteConversation(deleteTarget.id)}
              >
                删除
              </button>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
