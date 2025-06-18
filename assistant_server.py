import os
import time
from flask import Flask, request, jsonify, stream_with_context, Response
from dotenv import load_dotenv
from openai import OpenAI
import logging
import re

# --- Load environment variables ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('poker_assistant')

# --- Load all 10 assistant IDs from .env ---
ASSISTANT_IDS = {i: os.getenv(f"ASSISTANT_{i}") for i in range(1, 11)}

# --- OpenAI setup ---
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY missing")
    raise RuntimeError("OPENAI_API_KEY missing")
client = OpenAI(api_key=api_key)

app = Flask(__name__)

# --- Assistant Routing Logic ---

def select_assistant(game_state: dict) -> int:
    """Route to correct assistant STRICTLY according to assistant_tags only."""
    tags = game_state.get("assistant_tags", {})

    if not isinstance(tags, dict) or not any(tags.values()):
        raise RuntimeError(
            "assistant_tags missing or no valid tag is True (cannot select assistant)"
        )

    if tags.get("icm_spot"):
        return 2
    if tags.get("exploit_spot"):
        return 3
    if tags.get("preflop"):
        return 1
    if tags.get("pot_odds_relevant"):
        return 5
    if tags.get("spr_sensitive"):
        return 6
    if tags.get("bluff_catch_scenario"):
        return 7
    if tags.get("ev_delta_available"):
        return 8
    if tags.get("image_shift_active"):
        return 9
    if tags.get("overbet_active"):
        return 10

    raise RuntimeError("No assistant tag is set to True (cannot select assistant)")

def format_poker_prompt(game_state: dict) -> str:
    street = game_state.get("street", "Unknown").capitalize()
    position = game_state.get("position", "unknown")
    hero_stack = game_state.get("hero_stack", "unknown")
    bb_stack = game_state.get("bb_stack", "unknown")
    action = game_state.get("action", "")
    pot = game_state.get("pot", "unknown")
    board = " ".join(game_state.get("board_cards", []))
    villain_action = game_state.get("villain_action", "")
    available = [a.upper() for a in game_state.get("available_actions", ["FOLD", "CALL", "RAISE"])]

    prompt = f"{street}. You are in the {position} with {hero_stack}BB."
    if bb_stack != "unknown":
        prompt += f" The big blind has {bb_stack}BB"
    if villain_action:
        prompt += f" and has {villain_action}"
    if action:
        prompt += f" and has {action}"
    if pot != "unknown":
        prompt += f". The pot is {pot}BB."
    if board and street != "Preflop":
        prompt += f" Board: {board}."
    prompt += f" What’s the optimal decision?\n\n"
    prompt += f"Respond with only one recommendation: {', '.join(available)}"
    if "RAISE" in available:
        prompt += " (include amount if applicable)"
    prompt += ". Do not explain."
    return prompt

def normalize_result(raw_text: str) -> str:
    txt = raw_text.strip().upper()
    if "FOLD" in txt:
        return "RECOMMEND: FOLD"
    if "CALL" in txt:
        return "RECOMMEND: CALL"
    m = re.search(r'RAISE\s*(TO)?\s*([\d\.]+)', txt)
    if "RAISE" in txt and m:
        return f"RECOMMEND: RAISE to {m.group(2)}"
    if "RAISE" in txt:
        return "RECOMMEND: RAISE"
    return f"RECOMMEND: UNKNOWN - {raw_text.strip()}"

# --- Main Route ---

@app.route("/api/advice", methods=["POST"])
def route_ocr_decision():
    import traceback
    t0 = time.time()
    try:
        game_state = request.get_json(force=True, silent=True) or {}
        logger.info(f"Received game_state: {game_state}")

        assistant_id = select_assistant(game_state)
        assistant_key = ASSISTANT_IDS.get(assistant_id)
        if not assistant_key:
            raise RuntimeError(f"Assistant {assistant_id} not configured in .env")

        prompt = format_poker_prompt(game_state)
        logger.info(
            f"Routing to assistant {assistant_id}: {prompt.replace(chr(10),' ')}"
        )

        def generate():
            messages = [
                {"role": "system", "content": f"You are {assistant_key}."},
                {"role": "user", "content": prompt}
            ]
            try:
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=messages,
                    max_tokens=50,
                    temperature=0.2,
                    stream=True,
                )
                buffer = ""
                for chunk in response:
                    content = getattr(chunk.choices[0].delta, "content", None)
                    if content:
                        buffer += content
                        yield content
                yield f"\n\n<!-- {normalize_result(buffer)} -->"
            except Exception as e:
                logger.error(f"OpenAI error: {e}\n{traceback.format_exc()}")
                yield f"RECOMMEND: UNKNOWN - {str(e)}"

        return Response(stream_with_context(generate()), mimetype="text/plain")
    except Exception as e:
        logger.error("Exception in /api/advice: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run("0.0.0.0", 5000, debug=True)
