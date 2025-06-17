# Prompts for Further Development via LLM

The following prompts can be provided to an advanced language model to identify improvements, disruptive innovations, or fixes for `assistant_server.py`.

1. **Security Hardening**
   - "Review the `assistant_server.py` code and suggest comprehensive improvements to secure the API against common web vulnerabilities (e.g., injection, brute force attacks). Provide detailed code examples."

2. **OCR Accuracy Enhancements**
   - "Analyze the current OCR approach using `pytesseract` and `easyocr`. Propose optimized techniques or configurations to maximize accuracy and reduce latency."

3. **Scalability and Deployment**
   - "Design a scalable deployment strategy for the Flask API, considering containerization, orchestration, and monitoring. Outline the required code or configuration changes."

4. **Advanced GTO Matching**
   - "Suggest algorithms or data structures to improve the `get_gto_advice` function so it can handle complex game states and provide real-time advice with minimal latency."

5. **Extensibility**
   - "Identify sections of the code that would benefit from refactoring into separate modules or classes, enhancing maintainability. Provide example structures."

6. **Testing Strategy**
   - "Create a detailed plan for unit and integration tests for each function in `assistant_server.py`. Include mock inputs and expected outputs."

7. **GTO Chart v3 Integration**
   - "Modify `assistant_server.py` to parse the new `gto_chart.json` structure (version 3.0). Ensure `get_gto_advice` can access hand-strength categories, position data, and action mappings without introducing regressions. Provide clean, well-documented code."

8. **Efficient Lookups**
   - "Propose data structures or caching strategies to quickly retrieve advice from large GTO charts. The solution should minimize latency when matching game state fields such as pot size, position, and situation type."

