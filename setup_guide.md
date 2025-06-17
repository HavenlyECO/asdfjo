# Poker Assistant Setup Guide

## Cloud Server

1. **Prerequisites**
   - Python 3.8+
   - Install [EasyOCR](https://github.com/JaidedAI/EasyOCR) requirements if using GPU acceleration (optional).

2. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Configure Server**
   - Create a `.env` file in the project root with the following variables:
     - `POKER_ASSISTANT_API_KEY`
     - `POKER_ASSISTANT_ALLOWED_IPS`
    - `ASSISTANT_1` (and additional IDs as needed)
     - `OPENAI_API_KEY`
   - Place your `gto_chart.json` in the same directory.

4. **Run the Server**
   ```
   python assistant_server.py
   ```
   - For production: use `gunicorn` and HTTPS (see below).

5. **Enable HTTPS (DigitalOcean)**
   - Set up Nginx reverse proxy and use [Certbot](https://certbot.eff.org/) for Let's Encrypt.
   - Or run Flask with SSL certs (less secure, not recommended).

## Local Thin Client

1. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Configure Client**
   - Edit `local_client.py`:
     - Set `SERVER_URL` to server address.
     - Set `API_KEY` to match the server.
     - Adjust `CAPTURE_REGION` to fit your poker table on screen.

3. **Run the Client**
   ```
   python local_client.py
   ```

## Security Tips

- Always run the server behind HTTPS.
- Use strong, unique API keys.
- Restrict allowed IPs on the server.
- Never store sensitive information on the client.

## Scaling/Improvements

- **Dockerize** the Flask server for easy deployment.
- Replace GTO chart lookup with a proper solver or ML model.
- Support multiple simultaneous tables by running several screen regions.
- Add hotkey activation to the client for manual capture.
- Use a database for advanced statistics/logging.
