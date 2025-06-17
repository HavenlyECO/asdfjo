# Server Settings Storage Plan

This repository expects configuration values such as API keys and allowed IP addresses to be stored in a `.env` file. To keep sensitive data safe:

1. **Use `.env.template` as a template.** Copy this file to `.env` and replace each placeholder value with your actual keys.
2. **Keep `.env` out of version control.** The `.gitignore` file already excludes `.env` so secrets are not accidentally committed.
3. **Restrict permissions.** Limit read access to `.env` on the server (e.g., `chmod 600 .env`).
4. **Avoid storing secrets in client code.** Only the server should contain the real API keys. The client should reference a separate key meant for client authentication.
5. **Rotate keys when needed.** If you suspect a key was exposed, generate a new one and update `.env` accordingly.

Following these guidelines will help ensure that environment variables remain secure.
