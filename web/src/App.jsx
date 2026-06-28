import { useEffect, useMemo, useRef, useState } from "react";
import { uid } from "./utils";
import {
  clearRuntimeLog as clearRemoteRuntimeLog,
  fetchBootstrap,
  fetchCommands,
  fetchMemory,
  fetchProjects,
  fetchRuntimeLog,
  createConversation as createRemoteConversation,
  deleteConversation as deleteRemoteConversation,
  deleteProject as deleteRemoteProject,
  pickProject as pickRemoteProject,
  saveMemory as saveRemoteMemory,
  selectProject as selectRemoteProject,
  sendChat,
  cancelGeneration,
  applyEvent,
} from "./api";
import Message from "./components/Message";
import RuntimeStatus from "./components/RuntimeStatus";
import CommandMenu from "./components/CommandMenu";

const LEGACY_STORAGE_KEY = "agent-runtime-web-state-v1";
const ACTIVE_CONVERSATION_KEY = "agent-runtime-active-conversation-v1";
const ACTIVE_PROJECT_KEY = "agent-runtime-active-project-v1";
const REASONING_ENABLED_KEY = "agent-runtime-reasoning-enabled-v1";
const BOTTOM_FOLLOW_THRESHOLD_PX = 120;

function isNearBottom(element) {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= BOTTOM_FOLLOW_THRESHOLD_PX;
}

function normalizeConversation(conversation) {
  return {
    id: conversation.id || uid(),
    title: conversation.title || "新对话",
    messages: Array.isArray(conversation.messages)
      ? conversation.messages.map(normalizeMessage)
      : [],
    createdAt: conversation.createdAt || Date.now(),
    updatedAt: conversation.updatedAt || conversation.createdAt || Date.now(),
    context: conversation.context ?? "",
    contextUsed: conversation.contextUsed ?? 0,
    inputBudgetTokens: conversation.inputBudgetTokens ?? null,
    contextWindow: conversation.contextWindow ?? null,
  };
}

function normalizeProject(project) {
  return {
    id: project.id || uid(),
    name: project.name || "未命名项目",
    path: project.path || "",
    createdAt: project.createdAt || Date.now(),
    updatedAt: project.updatedAt || project.createdAt || Date.now(),
    lastOpenedAt: project.lastOpenedAt || 0,
    conversationCount: project.conversationCount ?? 0,
  };
}

function normalizeMessage(message) {
  return {
    id: message.id || uid(),
    role: message.role,
    text: message.text ?? message.content ?? "",
    noticeText: message.noticeText ?? "",
    reasoning: message.reasoning,
    reasoningOpen: Boolean(message.reasoningOpen),
    reasoningComplete: Boolean(message.reasoningComplete),
    streamComplete: Boolean(message.streamComplete),
    completedKey: message.completedKey ?? "",
    steps: Array.isArray(message.steps) ? message.steps : [],
    tools: Array.isArray(message.tools) ? message.tools : [],
    tone: message.tone,
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

function RuntimeLogModal({ log, loading, error, onRefresh, onClear, onClose }) {
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
            <button
              type="button"
              className="danger"
              onClick={onClear}
              disabled={loading}
            >
              清空
            </button>
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

function MemoryModal({
  memory,
  draft,
  loading,
  error,
  notice,
  onDraftChange,
  onSave,
  onClose,
}) {
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
        className="runtimeLogDialog memoryDialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="memory-title"
      >
        <header>
          <div>
            <h2 id="memory-title">长期记忆</h2>
            <p>{memory.lineCount ? `${memory.lineCount} 行记忆` : "暂无记忆"}</p>
          </div>
          <div className="runtimeLogActions">
            <button
              type="button"
              className="primary"
              onClick={onSave}
              disabled={loading}
            >
              保存
            </button>
            <button type="button" onClick={onClose}>关闭</button>
          </div>
        </header>
        {notice && <div className="runtimeLogNotice">{notice}</div>}
        {error && <div className="runtimeLogError">{error}</div>}
        <div className="memoryBody">
          <label className="memoryEditor">
            <span className="memoryLabel">当前记忆</span>
            <textarea
              value={draft}
              onChange={(event) => onDraftChange(event.target.value)}
              placeholder="每行一条长期记忆。保存会替换当前全部记忆。"
              spellCheck={false}
            />
          </label>
        </div>
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

function ProjectChooser({
  projects,
  error,
  onOpenProject,
  onSelect,
}) {
  return (
    <div className="projectChooser">
      <div className="projectChooserInner">
        <h1>选择项目</h1>
        <button type="button" className="projectOpenButton" onClick={onOpenProject}>
          打开项目
        </button>
        <div className="projectGrid">
          {projects.map((project) => (
            <button
              type="button"
              className="projectCard"
              key={project.id}
              onClick={() => onSelect(project.id)}
            >
              <span className="projectName">{project.name}</span>
              <span className="projectPath">{project.path}</span>
            </button>
          ))}
        </div>
        {projects.length === 0 && <p>没有可用项目。</p>}
        {error && <div className="error">{error}</div>}
      </div>
    </div>
  );
}

function ProjectEmpty({ project, error, onCreateConversation }) {
  return (
    <div className="projectEmpty">
      <div className="projectEmptyInner">
        <div className="projectEmptyLabel">当前项目</div>
        <h1>{project.name}</h1>
        <p>{project.path}</p>
        <button type="button" onClick={onCreateConversation}>
          新对话
        </button>
        {error && <div className="error">{error}</div>}
      </div>
    </div>
  );
}

export default function App() {
  const [conversationState, setConversationState] = useState(() => {
    return { items: [], activeId: "" };
  });
  const [projects, setProjects] = useState([]);
  const [activeProjectId, setActiveProjectId] = useState("");
  const [expandedProjectIds, setExpandedProjectIds] = useState([]);
  const [input, setInput] = useState("");
  const [promptHistory, setPromptHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(null);
  const [status, setStatus] = useState({
    cwd: "agent",
    cwdPath: "",
    model: "未配置",
    context: "",
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
  const [activeRequests, setActiveRequests] = useState({});
  const [error, setError] = useState("");
  const [commandPanel, setCommandPanel] = useState(null);
  const [activeCommandIndex, setActiveCommandIndex] = useState(0);
  const [commandMenuDismissed, setCommandMenuDismissed] = useState(false);
  const [conversationsLoaded, setConversationsLoaded] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [projectDeleteTarget, setProjectDeleteTarget] = useState(null);
  const [showJumpToBottom, setShowJumpToBottom] = useState(false);
  const [logModalOpen, setLogModalOpen] = useState(false);
  const [runtimeLog, setRuntimeLog] = useState({
    path: "",
    content: "",
    exists: false,
  });
  const [runtimeLogLoading, setRuntimeLogLoading] = useState(false);
  const [runtimeLogError, setRuntimeLogError] = useState("");
  const [memoryModalOpen, setMemoryModalOpen] = useState(false);
  const [memoryState, setMemoryState] = useState({
    content: "",
    lineCount: 0,
  });
  const [memoryDraft, setMemoryDraft] = useState("");
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [memoryError, setMemoryError] = useState("");
  const [memoryNotice, setMemoryNotice] = useState("");
  const [memorySaveNoticeOpen, setMemorySaveNoticeOpen] = useState(false);
  const [memoryCloseConfirmOpen, setMemoryCloseConfirmOpen] = useState(false);
  const transcriptRef = useRef(null);
  const textareaRef = useRef(null);
  const controllersRef = useRef(new Map());
  const autoFollowRef = useRef(true);

  const conversations = conversationState.items;
  const activeProject =
    projects.find((project) => project.id === activeProjectId) ?? null;
  const activeConversation =
    conversations.find((conversation) => conversation.id === conversationState.activeId) ?? null;
  const activeMessages = activeConversation?.messages ?? [];
  const activeRequest = activeConversation ? activeRequests[activeConversation.id] : null;

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
      localStorage.setItem(ACTIVE_PROJECT_KEY, activeProjectId);
    } catch {
      // localStorage may be unavailable in restricted browser contexts.
    }
  }, [activeProjectId, conversationState.activeId, conversationsLoaded]);

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
    setStatus((current) => {
      const nextContext = conversation.context ?? current.context;
      const nextUsed = conversation.contextUsed ?? current.contextUsed;
      const nextBudget = conversation.inputBudgetTokens ?? current.inputBudgetTokens;
      const nextWindow = conversation.contextWindow ?? current.contextWindow;
      if (
        nextContext === current.context &&
        nextUsed === current.contextUsed &&
        nextBudget === current.inputBudgetTokens &&
        nextWindow === current.contextWindow
      ) {
        return current;
      }
      return {
        ...current,
        context: nextContext,
        contextUsed: nextUsed,
        inputBudgetTokens: nextBudget,
        contextWindow: nextWindow,
      };
    });
  }

  useEffect(() => {
    let mounted = true;
    Promise.all([fetchBootstrap(), fetchCommands(), fetchProjects()])
      .then(([boot, commandData, projectData]) => {
        const projectItems = (projectData.items ?? []).map(normalizeProject);
        if (!mounted) return;
        setCommands(commandData.commands ?? []);
        setProjects(projectItems);
        setPromptHistory([]);
        setConversationState({ items: [], activeId: "" });
        setActiveProjectId("");
        setConversationsLoaded(true);
        for (const event of boot.events ?? []) {
          if (event.type === "status") setStatus((current) => ({ ...current, ...event }));
          if (event.type === "notice") setError(event.text ?? "");
        }
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
  }, [activeMessages, activeRequest]);

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
    if (!activeConversation) return;
    await submitText(input.trim(), activeConversation.id);
  }

  async function submitText(value, conversationId) {
    if (!value || activeRequests[conversationId]) return;
    autoFollowRef.current = true;
    setShowJumpToBottom(false);
    requestAnimationFrame(() => scrollTranscriptToBottom());
    const requestId = uid();
    const controller = new AbortController();
    controllersRef.current.set(conversationId, controller);
    setInput("");
    setError("");
    setCommandMenuDismissed(false);
    setActiveRequests((current) => ({
      ...current,
      [conversationId]: { requestId },
    }));
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
      if (err.name !== "AbortError") {
        if (isCommand) {
          setCommandPanel((current) => {
            if (!current) return current;
            return { ...current, tone: "error", lines: [...current.lines, String(err)] };
          });
        } else {
          setError(String(err));
        }
      }
    } finally {
      controllersRef.current.delete(conversationId);
      setActiveRequests((current) => {
        if (current[conversationId]?.requestId !== requestId) return current;
        const next = { ...current };
        delete next[conversationId];
        return next;
      });
    }
  }

  async function stopGeneration() {
    if (!activeConversation) return;
    const conversationId = activeConversation.id;
    const requestId = activeRequests[conversationId]?.requestId;
    if (!requestId) return;
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
      controllersRef.current.get(conversationId)?.abort();
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
    if (!activeConversation) return;
    submitText(name, activeConversation.id);
  }

  function updateProjectConversationCount(projectId, updater) {
    setProjects((current) =>
      current.map((project) => {
        if (project.id !== projectId) return project;
        const nextCount = Math.max(0, updater(project.conversationCount ?? 0));
        return { ...project, conversationCount: nextCount };
      }),
    );
  }

  function toggleProject(projectId) {
    if (projectId !== activeProjectId) {
      selectProject(projectId);
      return;
    }
    setExpandedProjectIds((current) =>
      current.includes(projectId)
        ? current.filter((id) => id !== projectId)
        : [...current, projectId],
    );
  }

  async function selectProject(projectId) {
    setError("");
    try {
      const payload = await selectRemoteProject(projectId);
      const selectedProject = normalizeProject(payload.project ?? {});
      const items = (payload.conversations ?? []).map(normalizeConversation);
      const selectedWithCount = {
        ...selectedProject,
        conversationCount:
          typeof payload.project?.conversationCount === "number"
            ? payload.project.conversationCount
            : items.length,
      };
      setProjects((current) =>
        current.some((project) => project.id === selectedWithCount.id)
          ? current.map((project) =>
              project.id === selectedWithCount.id ? selectedWithCount : project,
            )
          : [selectedWithCount, ...current],
      );
      setActiveProjectId(selectedWithCount.id);
      setExpandedProjectIds((current) =>
        current.includes(selectedWithCount.id) ? current : [...current, selectedWithCount.id],
      );
      setConversationState({ items, activeId: items[0]?.id ?? "" });
      setPromptHistory(promptsFromConversations(items));
      setInput("");
      setCommandMenuDismissed(false);
      applyConversationStatus(items[0] ?? null);
      if (items.length > 0) {
        requestAnimationFrame(() => document.querySelector(".composer textarea")?.focus());
      }
    } catch (err) {
      setError(String(err));
    }
  }

  async function openProject() {
    setError("");
    try {
      const payload = await pickRemoteProject();
      if (payload.cancelled) return;
      const project = normalizeProject(payload.project ?? {});
      setProjects((current) => [
        project,
        ...current.filter((item) => item.id !== project.id),
      ]);
      await selectProject(project.id);
    } catch (err) {
      setError(String(err));
    }
  }

  async function createNewConversation() {
    if (!activeProject) return;
    try {
      const conversation = normalizeConversation(
        await createRemoteConversation("新对话", activeProject.id),
      );
      setConversationState((current) => ({
        items: [conversation, ...current.items],
        activeId: conversation.id,
      }));
      setExpandedProjectIds((current) =>
        current.includes(activeProject.id) ? current : [...current, activeProject.id],
      );
      updateProjectConversationCount(activeProject.id, (count) => count + 1);
      setInput("");
      setError("");
      setCommandMenuDismissed(false);
      applyConversationStatus(conversation);
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
    setConversationState((current) => {
      const remaining = current.items.filter((conversation) => conversation.id !== conversationId);
      if (remaining.length === 0) {
        return { items: [], activeId: "" };
      }
      const activeId = current.activeId === conversationId ? remaining[0].id : current.activeId;
      return { items: remaining, activeId };
    });
    if (activeProject) {
      updateProjectConversationCount(activeProject.id, (count) => count - 1);
    }
  }

  async function deleteProject(projectId) {
    try {
      await deleteRemoteProject(projectId);
    } catch (err) {
      setError(String(err));
      return;
    }
    setProjectDeleteTarget(null);
    setProjects((current) => current.filter((project) => project.id !== projectId));
    setExpandedProjectIds((current) => current.filter((id) => id !== projectId));
    if (projectId === activeProjectId) {
      setActiveProjectId("");
      setConversationState({ items: [], activeId: "" });
      setPromptHistory([]);
      setInput("");
      setCommandMenuDismissed(false);
    }
    setError("");
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

  async function clearRuntimeLog() {
    setRuntimeLogLoading(true);
    setRuntimeLogError("");
    try {
      const payload = await clearRemoteRuntimeLog();
      setRuntimeLog({
        path: payload.path || "",
        content: "",
        exists: Boolean(payload.exists),
      });
      if (payload.error) setRuntimeLogError(payload.error);
    } catch (err) {
      setRuntimeLogError(String(err));
    } finally {
      setRuntimeLogLoading(false);
    }
  }

  async function openMemory() {
    setMemoryModalOpen(true);
    setMemoryNotice("");
    setMemorySaveNoticeOpen(false);
    setMemoryCloseConfirmOpen(false);
    await refreshMemory();
  }

  function applyMemoryPayload(payload) {
    const content = payload.content || "";
    setMemoryState({
      content,
      lineCount: Number(payload.lineCount ?? (content ? content.split(/\r?\n/).length : 0)),
    });
    setMemoryDraft(content);
    if (payload.error) setMemoryError(payload.error);
  }

  async function refreshMemory() {
    setMemoryLoading(true);
    setMemoryError("");
    setMemoryNotice("");
    try {
      applyMemoryPayload(await fetchMemory());
    } catch (err) {
      setMemoryError(String(err));
    } finally {
      setMemoryLoading(false);
    }
  }

  async function saveMemory({ showNotice = true } = {}) {
    setMemoryLoading(true);
    setMemoryError("");
    setMemoryNotice("");
    try {
      applyMemoryPayload(await saveRemoteMemory(memoryDraft));
      setMemoryNotice("记忆已保存。");
      if (showNotice) setMemorySaveNoticeOpen(true);
      return true;
    } catch (err) {
      setMemoryError(String(err));
      return false;
    } finally {
      setMemoryLoading(false);
    }
  }

  function closeMemory() {
    if (memoryDraft !== memoryState.content) {
      setMemoryCloseConfirmOpen(true);
      return;
    }
    setMemoryModalOpen(false);
    setMemoryNotice("");
    setMemoryError("");
  }

  function discardMemoryChanges() {
    setMemoryDraft(memoryState.content);
    setMemoryCloseConfirmOpen(false);
    setMemoryModalOpen(false);
    setMemoryNotice("");
    setMemoryError("");
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="sideHeader">
          <div>
            <div className="brand">智能体运行时</div>
          </div>
          <div className="sideActions">
            <button type="button" onClick={openProject}>
              打开项目
            </button>
            <button type="button" onClick={createNewConversation} disabled={!activeProject}>
              新对话
            </button>
          </div>
        </div>
        <nav className="conversationList" aria-label="项目和对话列表">
          <div className="sidebarSectionTitle">项目</div>
          {projects.map((project) => (
            <div className="projectTreeNode" key={project.id}>
              <div className="projectListRow">
                <button
                  type="button"
                  className={`projectListItem ${project.id === activeProjectId ? "active" : ""}`}
                  onClick={() => toggleProject(project.id)}
                  aria-expanded={expandedProjectIds.includes(project.id)}
                >
                  <span>{project.name}</span>
                  <span className="projectCount">{project.conversationCount ?? 0}</span>
                </button>
                <button
                  type="button"
                  className="projectDelete"
                  aria-label={`移除项目 ${project.name}`}
                  onClick={() => setProjectDeleteTarget(project)}
                >
                  ×
                </button>
              </div>
              {project.id === activeProjectId && expandedProjectIds.includes(project.id) && (
                <div className="projectBranch">
                  <div className="projectBranchTitle">对话</div>
                  {conversations.map((conversation) => (
                    <div
                      key={conversation.id}
                      className={`conversationItem ${conversation.id === activeConversation?.id ? "active" : ""} ${activeRequests[conversation.id] ? "running" : ""}`}
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
                </div>
              )}
            </div>
          ))}
        </nav>
      </aside>

      <section className="chatPane">
        {!activeProject ? (
          <ProjectChooser
            projects={projects}
            error={error}
            onOpenProject={openProject}
            onSelect={selectProject}
          />
        ) : !activeConversation ? (
          <ProjectEmpty
            project={activeProject}
            error={error}
            onCreateConversation={createNewConversation}
          />
        ) : (
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
            {activeRequest && (
              <TypingIndicator />
            )}
          </section>
        )}
        {activeProject && activeConversation && showJumpToBottom && (
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

        {activeProject && activeConversation && (
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
            {!commandPanel && error && <div className="error">{error}</div>}
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
                className={activeRequest ? "stop" : ""}
                onClick={activeRequest ? stopGeneration : submit}
                disabled={!activeRequest && !input.trim()}
              >
                {activeRequest ? "停止" : "发送"}
              </button>
            </div>
            <RuntimeStatus status={status} />
          </div>
        </section>
        )}
      </section>
      <div className="sideUtilityButtons">
        <button
          type="button"
          className="runtimeLogButton"
          onClick={openRuntimeLog}
          title="查看运行日志"
          aria-label="查看运行日志"
        >
          日志
        </button>
        <button
          type="button"
          className="runtimeLogButton"
          onClick={openMemory}
          title="查看和管理长期记忆"
          aria-label="查看和管理长期记忆"
        >
          记忆
        </button>
      </div>
      {logModalOpen && (
        <RuntimeLogModal
          log={runtimeLog}
          loading={runtimeLogLoading}
          error={runtimeLogError}
          onRefresh={refreshRuntimeLog}
          onClear={clearRuntimeLog}
          onClose={() => setLogModalOpen(false)}
        />
      )}
      {memoryModalOpen && (
        <MemoryModal
          memory={memoryState}
          draft={memoryDraft}
          loading={memoryLoading}
          error={memoryError}
          notice={memoryNotice}
          onDraftChange={setMemoryDraft}
          onSave={saveMemory}
          onClose={closeMemory}
        />
      )}
      {memoryCloseConfirmOpen && (
        <div className="modalBackdrop confirmOverlay" role="presentation">
          <section
            className="confirmDialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="memory-close-dialog-title"
          >
            <h2 id="memory-close-dialog-title">保存记忆修改？</h2>
            <p>当前记忆有未保存修改，关闭后会丢失这些内容。</p>
            <div className="confirmActions">
              <button type="button" onClick={() => setMemoryCloseConfirmOpen(false)}>
                继续编辑
              </button>
              <button type="button" onClick={discardMemoryChanges}>
                不保存
              </button>
              <button
                type="button"
                className="primary"
                onClick={async () => {
                  if (await saveMemory({ showNotice: false })) {
                    setMemoryCloseConfirmOpen(false);
                    setMemoryModalOpen(false);
                  }
                }}
              >
                保存并关闭
              </button>
            </div>
          </section>
        </div>
      )}
      {memorySaveNoticeOpen && (
        <div className="modalBackdrop confirmOverlay" role="presentation">
          <section
            className="confirmDialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="memory-save-dialog-title"
          >
            <h2 id="memory-save-dialog-title">保存成功</h2>
            <p>长期记忆已保存。</p>
            <div className="confirmActions">
              <button
                type="button"
                className="primary"
                onClick={() => setMemorySaveNoticeOpen(false)}
              >
                知道了
              </button>
            </div>
          </section>
        </div>
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
      {projectDeleteTarget && (
        <div className="modalBackdrop" role="presentation">
          <section
            className="confirmDialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="project-delete-dialog-title"
          >
            <h2 id="project-delete-dialog-title">移除项目？</h2>
            <p>
              {projectDeleteTarget.name}
              <br />
              只从项目列表移除，不会删除本机文件。
            </p>
            <div className="confirmActions">
              <button type="button" onClick={() => setProjectDeleteTarget(null)}>
                取消
              </button>
              <button
                type="button"
                className="danger"
                onClick={() => deleteProject(projectDeleteTarget.id)}
              >
                移除
              </button>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
