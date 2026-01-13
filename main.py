"""Salty meeting notes assistant."""

import logging
import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

import salty

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("salty").setLevel(logging.DEBUG)

def send_slack(text: str) -> dict:
    """Send a message to Slack."""
    slack_hook = os.getenv("SLACK_HOOK")

    if not slack_hook:
        return {"error": "SLACK_HOOK environment variable not set"}

    try:
        response = requests.post(
            slack_hook,
            json={"text": text},
            timeout=10,
        )
        response.raise_for_status()
        return {"success": True, "message": f"Sent to Slack: {text}"}
    except requests.RequestException as e:
        return {"error": f"Failed to send to Slack: {str(e)}"}


send_slack_tool = salty.Tool(
    name="send_slack",
    description="Send a message to Slack",
    function=send_slack,
    parameters={
        "text": {
            "type": "string",
            "description": "The message to send to Slack",
        }
    },
    required=["text"],
)


# Build the app
app = salty.build(
    instructions="""You are Salty, a friendly Reachy Mini robot assistant for meetings.

Your job is to help take notes during meetings. When someone says something important
that should be written down, use the send_slack tool to record it whenever a point is
made or whe the user asks you.

Be proactive - if you hear action items, decisions, or important points, offer to save them.
Use the express_emotion tool to show engagement and emotion during conversations.
Finally, mute and unmute yourself as needed. When the user says for you to stop talking
mute yourself until the user calls you back.
""",
    voice="cedar",
    api_key=None,
    tools=[send_slack_tool],
)


if __name__ == "__main__":
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
