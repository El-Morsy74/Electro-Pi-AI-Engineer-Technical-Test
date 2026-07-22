import logging
import os
import time
import asyncio
from dotenv import load_dotenv

# LiveKit Core Framework Imports
from livekit import agents
from livekit.agents.beta.tools import EndCallTool
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    room_io,
    function_tool,
    RunContext,
    ToolError,
    AgentStateChangedEvent,
    MetricsCollectedEvent,
    metrics,
    llm,
    stt,
    tts,
    inference
)
from livekit.plugins import noise_cancellation, silero, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# ==============================================================================
# ENVIRONMENT & LOGGING SETUP
# ==============================================================================
# Load API keys and configurations from local .env file
load_dotenv()

# Configure module-level logger
logger = logging.getLogger(__name__)

# ==============================================================================
# MOCK DATABASE (Food Delivery Orders)
# ==============================================================================
MOCK_ORDERS = {
    "BB-1234": {
        "status": "Out for delivery",
        "eta": "10 minutes",
        "items": "1x Double Cheeseburger, 1x Large Fries, 1x Vanilla Milkshake",
        "total": "$18.50",
        "delivery_address": "123 Maple Street",
    },
    "BB-5678": {
        "status": "Preparing in kitchen",
        "eta": "25 minutes",
        "items": "2x Pepperoni Pizza, 1x Garlic Bread, 1x Diet Coke",
        "total": "$32.00",
        "delivery_address": "456 Oak Avenue",
    },
    "BB-9012": {
        "status": "Delivered",
        "eta": "Delivered 15 minutes ago",
        "items": "1x Spicy Chicken Wrap, 1x Caesar Salad, 1x Bottled Water",
        "total": "$14.75",
        "delivery_address": "789 Pine Road",
    }
}

# ==============================================================================
# VOICE AGENT CLASS & TOOLS DEFINITION
# ==============================================================================
class Assistant(Agent):
    """BiteBuddy Customer Support Assistant Agent.
    
    Handles voice interactions, food order status inquiries via function tools,
    and automatic call termination.
    """
    def __init__(self) -> None:
        super().__init__(
            # System prompt instructions designed specifically for speech interaction
            instructions=(
                "You are BiteBuddy, a friendly customer support voice assistant for the BiteBuddy food delivery app.\n\n"
                "OPERATIONAL WORKFLOW:\n"
                "1. GREETING PHASE: On the first turn (when the user says hello or contacts you), welcome them friendly to BiteBuddy and ask them to provide their order number (formatted as BB-XXXX).\n"
                "2. STATUS LOOKUP PHASE: When the user provides an order ID (even if spoken like 'BB 1234' or 'b b one two three four'), immediately call the 'get_order_status' tool. Do NOT repeat or welcome them again.\n"
                "3. RESPONSE PHASE: Once the tool returns the status information, summarize the status, items, and ETA in 1 to 2 short conversational sentences (e.g. 'I found your order! It is out for delivery and will arrive in 10 minutes.').\n"
                "4. ERROR HANDLING: If the order status lookup fails (the order does not exist), tell the user that the order was not found and ask them to provide the correct ID.\n"
                "5. DISMISSAL: When the user says goodbye or finishes, thank them and call the 'end_call' tool.\n\n"
                "RULES:\n"
                "- Keep every reply short (1 to 2 sentences max).\n"
                "- Speak clearly and conversationally. Do not use markdown, lists, or special symbols."
            ),
            # EndCallTool allows the LLM to hang up the WebRTC room call
            tools=[EndCallTool(delete_room=True)]
        )

    @function_tool()
    async def get_order_status(
        self,
        context: RunContext,
        order_id: str,
    ) -> dict:
        """Look up the real-time status of a food delivery order.
        
        Args:
            order_id: The unique order identifier, formatted as BB-XXXX (e.g. BB-1234).
        """
        # Normalize order ID (strip spaces, hyphens, and convert to uppercase)
        clean_id = order_id.replace(" ", "").replace("-", "").upper()
        if clean_id.startswith("BB") and len(clean_id) == 6:
            order_id = f"BB-{clean_id[2:]}"
        else:
            order_id = order_id.strip().upper()
            
        logger.info(f"Looking up order status for ID: {order_id}")
        
        # Verify if order exists in the mock database
        if order_id not in MOCK_ORDERS:
            # Raising ToolError informs the LLM gracefully so it can tell the user
            raise ToolError(f"Order ID '{order_id}' was not found in our database. Please verify the ID.")
            
        order_info = MOCK_ORDERS[order_id]
        return {
            "order_id": order_id,
            "status": order_info["status"],
            "eta": order_info["eta"],
            "items": order_info["items"],
            "total": order_info["total"],
        }

# ==============================================================================
# AGENT SERVER & RTC SESSION ENTRYPOINT
# ==============================================================================
server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Entrypoint function called whenever a user connects to a WebRTC room."""
    logger.info(f"Starting session for room: {ctx.room.name}")

    # Load Voice Activity Detection model (Silero VAD)
    vad = silero.VAD.load()

    # Configure the Voice Pipeline Session with STT, Fallback LLM, TTS, and VAD
    session = AgentSession(
        # STT (Speech-to-Text) Pipeline with Fallback (Deepgram -> AssemblyAI)
        stt=stt.FallbackAdapter(
            [
                inference.STT.from_model_string("deepgram/nova-3"),
                inference.STT.from_model_string("assemblyai/universal-streaming"),
            ]
        ),
        # LLM Pipeline with Fallback (Primary: Groq -> Fallback: OpenRouter Free Gemini)
        llm=llm.FallbackAdapter(
            [
                openai.LLM(
                    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                    base_url="https://api.groq.com/openai/v1",
                    api_key=os.getenv("GROQ_API_KEY"),
                ),
                openai.LLM(
                    model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash:free"),
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                ),
            ]
        ),
        # TTS (Text-to-Speech) Pipeline with Voice Fallback
        tts=tts.FallbackAdapter(
            [
                inference.TTS.from_model_string("cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
                inference.TTS.from_model_string("inworld/inworld-tts-1"),
            ]
        ),
        vad=vad,
        # Turn detection identifies natural pauses to prevent accidental interruptions
        turn_detection=MultilingualModel(),
        # Latency optimization: LLM starts generation while user is concluding speech
        preemptive_generation=True,
    )

    # ==========================================================================
    # METRICS COLLECTION & LATENCY TELEMETRY
    # ==========================================================================
    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        """Collect and log performance metrics (TTFT, TTS latency, EOU)."""
        nonlocal last_eou_metrics
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        """Calculate and log Time to First Audio (TTFA) when agent starts speaking."""
        if ev.new_state == "speaking":
            if last_eou_metrics:
                elapsed = time.time() - last_eou_metrics.timestamp
                logger.info(f"--- Time to First Audio (TTFA): {elapsed:.3f}s ---")

    async def log_usage():
        """Shutdown callback to summarize total session usage."""
        summary = usage_collector.get_summary()
        logger.info(f"Session complete. Usage summary: {summary}")

    # Register session shutdown callback
    ctx.add_shutdown_callback(log_usage)

    # Start the real-time agent session in the WebRTC room
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),  # Background noise cancellation
            ),
        ),
    )

# ==============================================================================
# CLI EXECUTION ENTRYPOINT
# ==============================================================================
def main():
    """Run the LiveKit agent application via CLI."""
    agents.cli.run_app(server)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
