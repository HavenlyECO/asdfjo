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
    street = game_state.get("street", "").lower()
    position = game_state.get("position", "").lower()
    stack = game_state.get("hero_stack", 100)
    pot = game_state.get("pot", 0)
    villain_stats = game_state.get("villain_stats", {})
    action = game_state.get("action", "").lower()
    board = game_state.get("board_cards", [])
    spr = game_state.get("spr", None)
    image_shift = game_state.get("image_shift", False)
    overbet = game_state.get("overbet", False)
    icm_context = game_state.get("icm", False)
    is_bubble = game_state.get("bubble", False)

    # 2: Stack-Based ICM Assistant
    if icm_context or is_bubble or stack < 15:
        return 2
    # 3: Villain Exploit Assistant
    if villain_stats and any(stat in villain_stats for stat in ["vpip", "fold_to_3b", "pfr", "af"]):
        return 3
    # 4: Board Texture Assistant
    if street in ["flop", "turn", "river"] and board:
        return 4
    # 5: Pot Odds Assistant
    if "pot_odds" in game_state and game_state["pot_odds"] is not None:
        return 5
    # 6: Future Street Pressure Assistant
    if spr is not None and spr < 3:
        return 6
    # 7: Bluff Catcher Evaluator
    if street == "river" and "bluff" in action:
        return 7
    # 8: EV Delta Comparator Assistant
    if "ev_fold" in game_state or "ev_call" in game_state or "ev_raise" in game_state:
        return 8
    # 9: Meta-Image Shift Assistant
    if image_shift:
        return 9
    # 10: Overbet Detection Assistant
    if overbet or (street in ["turn", "river"] and "overbet" in action):
        return 10
    # 1: Preflop Position Assistant (default)
    if street == "preflop":
        return 1

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

@app.route("/api/poker_decision", methods=["POST"])
def route_ocr_decision():
    t0 = time.time()
    game_state = request.json or {}

    assistant_id = select_assistant(game_state)
    assistant_key = ASSISTANT_IDS.get(assistant_id)
    if not assistant_key:
        return jsonify({"error": f"Assistant {assistant_id} not configured"}), 500

    prompt = format_poker_prompt(game_state)
    logger.info(f"Routing to assistant {assistant_id}: {prompt.replace(chr(10),' ')}")

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
            # Yield normalized result as a comment for API clients
            yield f"\n\n<!-- {normalize_result(buffer)} -->"
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            yield f"RECOMMEND: UNKNOWN - {str(e)}"

    resp = Response(stream_with_context(generate()), mimetype="text/plain")
    resp.headers["X-Elapsed"] = str(time.time() - t0)
    return resp

if __name__ == "__main__":
    app.run("0.0.0.0", 5000, debug=True)
