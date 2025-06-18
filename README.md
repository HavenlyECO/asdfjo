# Poker Assistant

This project provides a Flask server and a local client for generating poker advice via OpenAI assistants.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.template` to `.env` and fill in your API keys and assistant IDs.
3. Run the server:
   ```bash
   python assistant_server.py
   ```
4. To quickly test the API with a sample game state, run `post_game_state.py`:
   ```bash
   python post_game_state.py
   ```
5. Optionally, use `local_client.py` to interact with the server from your desktop.

See `setup_guide.md` for detailed instructions.
