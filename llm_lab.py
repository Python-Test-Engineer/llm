"""llm_lab.py -- start the LLM Lab web app.

    uv run llm_lab.py

Then open the URL it prints (default http://127.0.0.1:8000/).

Set LLM_LAB_PORT to use a different port, LLM_LAB_HOST to expose it beyond
localhost. Everything runs from the Python standard library -- no installs.
"""

from lab.server import main

if __name__ == "__main__":
    main()
