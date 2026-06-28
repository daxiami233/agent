import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

function formatValue(value) {
  if (value === undefined) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function compactValue(value, limit = 160) {
  const text = formatValue(value).replace(/\s+/g, " ").trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, limit - 3)}...`;
}

function JsonView({ value }) {
  const json = formatValue(value);
  const pattern =
    /("(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*"(?=\s*:)|"(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*"|\b(?:true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;
  const nodes = [];
  let lastIndex = 0;
  let match;
  while ((match = pattern.exec(json)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(json.slice(lastIndex, match.index));
    }
    const token = match[0];
    let className = "jsonNumber";
    if (token.startsWith('"')) {
      className = json.slice(match.index + token.length).trimStart().startsWith(":")
        ? "jsonKey"
        : "jsonString";
    } else if (token === "true" || token === "false") {
      className = "jsonBoolean";
    } else if (token === "null") {
      className = "jsonNull";
    }
    nodes.push(
      <span className={className} key={`${match.index}-${token}`}>
        {token}
      </span>,
    );
    lastIndex = pattern.lastIndex;
  }
  if (lastIndex < json.length) nodes.push(json.slice(lastIndex));
  return <pre className="jsonView">{nodes}</pre>;
}

function normalizeMath(text) {
  return text
    .replace(/\\\[(.+?)\\\]/gs, "$$$$$1$$$$")
    .replace(/\\\((.+?)\\\)/g, "$$$1$$");
}

function MarkdownContent({ className = "", text }) {
  return (
    <div className={`markdownBody ${className}`.trim()}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
      >
        {normalizeMath(text)}
      </ReactMarkdown>
    </div>
  );
}

function ReasoningStep({ step, streamComplete }) {
  if (!step.text) return null;
  const isActive = !step.complete && !streamComplete;
  return (
    <details
      className="reasoning"
      open={isActive ? true : undefined}
      defaultOpen={Boolean(step.open) && !streamComplete}
    >
      <summary>
        <span>思考过程</span>
        <small>{step.complete || streamComplete ? "已完成" : "进行中"}</small>
      </summary>
      <div className="reasoningBody">
        <MarkdownContent className="reasoningMarkdown" text={step.text} />
      </div>
    </details>
  );
}

function ToolCard({ tool, streamComplete }) {
  const status = tool.status ?? "running";
  const isRunning = status === "running";
  const isError = status === "error";
  const isDenied = status === "denied";
  const isActive = isRunning && !streamComplete;
  return (
    <details
      className={`toolCard ${status}`}
      open={isActive ? true : undefined}
      defaultOpen={isActive}
    >
      <summary className="toolHeader">
        <span className="toolMeta">
          <strong>工具调用：{tool.name}</strong>
          <code>{tool.argumentsSummary || compactValue(tool.arguments)}</code>
        </span>
        <span className="toolStatus">
          {isRunning && <span className="spinner" aria-hidden="true" />}
          {isRunning ? "调用中" : isError ? "失败" : isDenied ? "拒绝" : "完成"}
        </span>
      </summary>
      {!isRunning && (
        <div className="toolResult">
          <JsonView value={tool.result ?? "无返回结果"} />
        </div>
      )}
    </details>
  );
}

function ExecutionSteps({ message }) {
  const steps = message.steps?.length ? message.steps : legacySteps(message);
  if (!steps.length) return null;
  return (
    <div className="executionSteps">
      {steps.map((step, index) => {
        const key = [
          step.type,
          step.id ?? index,
          step.status ?? "",
          step.complete ? "complete" : "active",
          step.open ? "open" : "closed",
          message.completedKey ?? "",
        ].join("-");
        if (step.type === "reasoning") {
          return (
            <ReasoningStep
              step={step}
              streamComplete={Boolean(message.streamComplete)}
              key={key}
            />
          );
        }
        if (step.type === "tool") {
          return (
            <ToolCard
              tool={step}
              streamComplete={Boolean(message.streamComplete)}
              key={key}
            />
          );
        }
        return null;
      })}
    </div>
  );
}

function legacySteps(message) {
  const steps = [];
  if (message.reasoning) {
    steps.push({
      id: "legacy-reasoning",
      type: "reasoning",
      text: message.reasoning,
      open: Boolean(message.reasoningOpen),
      complete: Boolean(message.reasoningComplete),
    });
  }
  for (const tool of message.tools ?? []) {
    steps.push({ ...tool, type: "tool" });
  }
  return steps;
}

export default function Message({ message }) {
  if (message.role === "notice") {
    return (
      <article className={`message notice ${message.tone ?? "muted"}`}>
        <div className="avatar">命令</div>
        <div className="content">
          <MarkdownContent className="noticeMarkdown" text={message.text} />
        </div>
      </article>
    );
  }

  return (
    <article className={`message ${message.role}`}>
      <div className="avatar">{message.role === "user" ? "我" : "助手"}</div>
      <div className="content">
        {message.role === "assistant" && <ExecutionSteps message={message} />}
        {message.text &&
          (message.role === "assistant" ? (
            <MarkdownContent className="assistantFinal" text={message.text} />
          ) : (
            <div>{message.text}</div>
          ))}
        {message.noticeText && (
          <div className="assistantNoticeSeparator">
            <MarkdownContent className="noticeMarkdown" text={message.noticeText} />
          </div>
        )}
      </div>
    </article>
  );
}
