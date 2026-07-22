# Section 1 Write-Up: Interruption Handling & Safe Tool Design

### 1. Barge-in & Interruption Handling
In real-time speech interactions, handling barge-in (user interrupting the agent) requires rapid synchronization between Voice Activity Detection (VAD) and downstream models:
- **VAD Triggering**: A local **Silero VAD** monitor processes incoming WebRTC audio. When user speech energy exceeds the threshold, it triggers a `user_started_speaking` event.
- **Buffer Flushing**: Upon detection, the agent session immediately signals the client WebRTC browser window to flush its output audio playout queue, cutting off ongoing Text-To-Speech (TTS) audio.
- **Task Cancellation**: Concurrently, the server cancels the running LLM token generation process and stops the TTS synthesis adapter to conserve compute resources.
- **Turn Detection Tuning**: We configure the Turn Detection model with a pause threshold (e.g., 600ms) to distinguish natural breathing breaks from completed thoughts, preventing accidental self-interruption.

### 2. Adding a Second Tool Safely
To safely add a new tool (e.g., address updates or cancellations) to the agent class:
- **Strict Schema Definitions**: Use Python type-hints (`str`, `int`) and explicit docstrings on the method decorated with `@function_tool()`. The LiveKit SDK auto-serializes this metadata into JSON Schema for the LLM to understand when and how to call it.
- **ToolError Boundary Wrapping**: Wrap all external API/database requests in try-except blocks. If validation fails (e.g., changing address for an order that is already shipped), raise a `ToolError("Friendly error message")`.
- **LLM Grounding**: LiveKit intercepts `ToolError` and passes it back to the LLM as standard tool feedback. The LLM can then explain the failure conversationally to the user (e.g., "I'm sorry, but your package is already out for delivery, so I cannot update the address.") instead of failing silently or hallucinating success.
