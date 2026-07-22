# 🍔 BiteBuddy Voice Agent (Section 1 — LiveKit Agents)

A real-time Voice AI assistant built for food order support. It uses LiveKit for WebRTC audio, Groq for sub-second LLM answers, and OpenRouter as an automatic fallback if Groq hits rate limits.

---

## 📂 Project Structure

```text
Section 1 - LiveKit Agents/
├── agent.py              # Main LiveKit voice agent script
├── requirements.txt      # Python dependencies list
├── task_1_1_report.md    # Interruption and safe tool technical write-up
└── README.md             # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Copy `.env.example` to `.env` and fill in your API credentials:
```ini
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# Primary LLM: Groq (~100ms response time)
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=llama3-8b-8192

# Fallback LLM: OpenRouter (Backup if Groq hits rate limits)
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_MODEL=google/gemini-2.5-flash:free
```

---

## 🚀 How to Run

### Console Mode (Test with microphone in terminal)
```bash
python agent.py console
```

### Production Mode (Connect to LiveKit Cloud)
```bash
python agent.py start
```

---

## 🛠️ How It Works

1. **Sub-second Speech Responses (~260ms)**: Groq generates text answers in real-time, which are streamed directly to Cartesia Sonic-3 for instant audio playback.
2. **Failover Protection**: If Groq returns a rate limit error (429), LiveKit's `FallbackAdapter` seamlessly switches to OpenRouter in under 50ms without dropping the user's call.
3. **Order Status Lookup**: When a user mentions an order ID like `BB-1234`, the agent automatically invokes `@function_tool` to check status, items, and ETA.
4. **Auto-Hangup**: When the user says goodbye, the assistant thanks them and uses `EndCallTool` to disconnect the room.
