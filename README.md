# Poker Assistant

This repository contains a full stack poker assistant that routes on-table information to specialized OpenAI assistants. It includes a Flask API, a high‑FPS client for capturing screenshots, and a collection of modules for computer vision and game analysis.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.template` to `.env` and fill in your OpenAI key and assistant IDs.
3. Run the server (set `ENABLE_YOLO=0` to disable card detection):
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

## Command Line Utilities

- `python assistant_server.py` – start the Flask API server. When running,
  every request saves a `logs/game_states/` snapshot (JSON + PNG) that can later
  be used with `train_card_yolo.py`. The server logs the exact paths so you can
  verify screenshots are being written. The `/api/advice` endpoint accepts
  either a screenshot file or a raw JSON game state.
- `python local_client.py` – capture the table region and show overlay advice.
- `python screenshot_server.py` – minimal server that stores uploaded screenshots.
- `python screenshot_saver.py` – continuously capture screenshots and upload them to the screenshot server.
- `python screenshot_capture_server.py` – serve on-demand screenshots to remote clients. Set `CAPTURE_SERVER_PORT` if a different port is required.
- `python screenshot_capture_client.py` – fetch screenshots from the capture server and display advice.
  The client loads environment variables (e.g., from a `.env` file); set `CAPTURE_SERVER_URL` to the capture server's full URL (e.g., `http://<host>:5002/api/capture`).
- `python post_game_state.py` – send a sample JSON game state to the server.
- `python realtime_pipeline.py` – run the demo multithreaded OCR pipeline.
- `python gto_proxy.py --train solver_data.json --out gto_proxy_model.pkl` – train the GTO decision tree model.
- `python hotkeys.py` – enable global F1/F2/F12 hotkeys.
- `python table_layout.py` – print computed table layout coordinates.
- `python frame_change_detector.py` – webcam test for frame change detection.
- `python equity_calculator.py` – run a quick Monte‑Carlo equity example.
- `python poker_hud.py` – launch the PyQt HUD demo.
- `python hand_logger.py` – log a few sample hands and export JSON/CSV.
- `python visual_confidence.py` – demo card/OCR confidence helpers.
- `python ocrcv_op.py` – simple end‑to‑end capture → OCR → recommendation loop.
- `python train_card_yolo.py --data card_data.yaml --out card_yolov8.pt` – train the YOLO card detection model.

See `setup_guide.md` for detailed configuration instructions.

## Repository Structure

- `assistant_server.py` – Flask API that streams advice from OpenAI.
- `local_client.py` – screenshot client with Tk overlay and optional TTS.
- `screenshot_capture_server.py` – Flask server that provides captured screenshots.
- `screenshot_capture_client.py` – thin client that relies on the capture server.
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
