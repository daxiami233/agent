# Context Compression Flow

```text
AgentLoop prepares a model request
    |
    v
ContextEngine.build_model_input(conversation_id, extra_input_tokens)
    |
    v
Load stored conversation messages
    |
    v
Convert StoredMessage records into ContextMessage records
    |
    v
_compress_if_needed(messages, extra_input_tokens)
    |
    v
Compute request_tokens
    |
    +--> request_tokens <= compact_threshold?
    |       |
    |       +-- yes ------------------------------------------------------+
    |                                                                  |
    |       +-- no                                                       |
    |           |                                                        |
    |           v                                                        |
    |   Split messages into:                                             |
    |       - one optional leading summary message                       |
    |       - raw conversation messages                                  |
    |           |                                                        |
    |           v                                                        |
    |   Split raw messages by user turns                                 |
    |           |                                                        |
    |           v                                                        |
    |   Keep the latest RECENT_TURNS turns as recent messages            |
    |   Treat the older turns as compressible history                    |
    |                                                                   |
    |   Turn-level compression range:                                    |
    |       [========== compress older turns ==========][ keep recent ]  |
    |       0%                                                        100%|
    |           |                                                        |
    |           v                                                        |
    |   older turns exist?                                               |
    |           |                                                        |
    |           +-- yes                                                  |
    |           |     |                                                  |
    |           |     v                                                  |
    |           | Compress existing summary + older turns                |
    |           | into one updated leading summary message               |
    |           |     |                                                  |
    |           |     v                                                  |
    |           | messages = updated summary + recent messages           |
    |           |                                                        |
    |           +-- no                                                   |
    |                 |                                                  |
    |                 v                                                  |
    |             messages = existing summary + recent messages          |
    |                                                                  |
    |           v                                                        |
    |   Recompute request_tokens                                         |
    |           |                                                        |
    |           v                                                        |
    |   request_tokens <= compact_threshold?                             |
    |           |                                                        |
    |           +-- yes --------------------------------------------------+
    |           |                                                        |
    |           +-- no                                                   |
    |                 |                                                  |
    |                 v                                                  |
    |         Compress older raw content inside the recent range          |
    |         using RAW_KEEP_RATIO                                       |
    |                                                                   |
    |         Recent-range compression target:                           |
    |             [====== compress raw prefix ======][ keep raw suffix ] |
    |             0%                                                  100%|
    |                 |                                                  |
    |                 v                                                  |
    |         Build atomic groups:                                       |
    |             - normal messages are one group                        |
    |             - assistant tool_calls and matching tool results       |
    |               stay in the same group                               |
    |                 |                                                  |
    |                 v                                                  |
    |         Keep the newest atomic-group suffix that fits the           |
    |         raw keep target                                            |
    |                 |                                                  |
    |                 v                                                  |
    |         Compress the older atomic-group prefix into the same        |
    |         leading summary message                                    |
    |                 |                                                  |
    |                 v                                                  |
    |         messages = updated summary + kept raw suffix                |
    |                 |                                                  |
    |                 v                                                  |
    |         Recompute request_tokens                                   |
    |                 |                                                  |
    |                 v                                                  |
    |         request_tokens <= input_budget?                            |
    |                 |                                                  |
    |                 +-- yes --------------------------------------------+
    |                 |                                                  |
    |                 +-- no                                             |
    |                       |                                            |
    |                       v                                            |
    |               Force truncate as the final fallback                 |
    |                                                                    |
    |               Final fallback keep range:                            |
    |                   [ drop ][ drop ][ keep newest atomic groups ]     |
    |                   0%                                           100% |
    |                       |                                            |
    |                       v                                            |
    |               Keep the newest atomic groups that fit               |
    |               input_budget; keep the summary only if it fits       |
    |                                                                    |
    +--------------------------------------------------------------------+
    |
    v
messages changed?
    |
    +-- yes
    |     |
    |     v
    | Replace stored conversation messages with compacted messages
    |
    +-- no
          |
          v
Return compacted messages
    |
    v
_repair_tool_call_sequence(messages)
    |
    v
Fix only abnormal historical tool-call structure:
    - orphan tool results become system fallback summaries
    - incomplete assistant tool_calls are preserved only when they are
      still at the tail and may be waiting for tool results
    |
    v
Build final model input:
    - main system prompt
    - optional leading compression summary
    - remaining recent raw messages
    |
    v
_raise_if_over_budget(model_input, extra_input_tokens)
    |
    +--> used_tokens <= input_budget?
            |
            +-- yes --> Return model_input to AgentLoop
            |
            +-- no  --> Raise ContextOverflowError and do not call provider
```
