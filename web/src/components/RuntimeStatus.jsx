import { tail } from "../utils";

export default function RuntimeStatus({ status }) {
  const contextWindow = Number(status.inputBudgetTokens);
  const contextUsed = Number(status.contextUsed ?? 0);
  const hasWindow = Number.isFinite(contextWindow) && contextWindow > 0;
  const contextPercent = hasWindow
    ? Math.min(100, Math.max(0, (contextUsed / contextWindow) * 100))
    : 0;

  return (
    <div className="runtimeStatus">
      <div className="runtimePath" title={status.cwdPath || status.cwd}>
        <strong>工作目录</strong>
        <span>{status.cwdPath || status.cwd}</span>
      </div>
      <div className="runtimeItems">
        <span className="runtimeItem model" title={status.model}>
          <strong>模型</strong>
          <span>{tail(status.model, 54)}</span>
        </span>
        <span className="runtimeItem context" title={status.context}>
          <strong>输入上下文</strong>
          <span>{status.context.replace(/^输入上下文：/, "")}</span>
        </span>
      </div>
      <div className="contextMeter" aria-label={status.context}>
        <div style={{ width: `${contextPercent}%` }} />
      </div>
    </div>
  );
}
