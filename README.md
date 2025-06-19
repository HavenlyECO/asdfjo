# Poker Assistant

This repository contains a full stack poker assistant that routes on-table information to specialized OpenAI assistants. It includes a Flask API, a high‑FPS client for capturing screenshots, and a collection of modules for computer vision and game analysis.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.template` to `.env` and fill in your OpenAI key and assistant IDs.
3. Run the server:
   ```bash
   python assistant_server.py
   ```
4. To post a sample game state directly, run `post_game_state.py`:
   ```bash
   python post_game_state.py
   ```
5. Optionally start the thin client to capture a table region and display advice:
   ```bash
   python local_client.py
   ```

See `setup_guide.md` for detailed configuration instructions.

## Repository Structure

- `assistant_server.py` – Flask API that streams advice from OpenAI.
- `local_client.py` – screenshot client with Tk overlay and optional TTS.
- `decision_engine.py` – combines solver output, LLM responses, equity calcs, and ICM logic.
- `gto_proxy.py` – fast decision tree proxy with optional pre-flop chart.
- `hero_turn_detector.py` – determines when it's your turn to act using screen cues.
- `card_detector.py` and `ocr_parser.py` – detect cards and parse text from screenshots.
- `villain_profiler.py` and `villain_exploit.py` – track opponent tendencies and adjust recommendations.
- `quality_monitor.py` – sanity-checks parsed game states.

## Development

Prompts for further code improvements are collected in `LLM_PROMPTS.md`. These can be provided to a language model to generate new ideas or disruptive optimizations.

## License

This project is provided as-is for educational purposes.
