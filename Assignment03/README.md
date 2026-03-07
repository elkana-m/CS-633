# Terminal Chatbot (Django / ChatterBot)

A terminal client that lets you chat with a bot using Django and ChatterBot. The bot is a machine-learning based conversational dialog engine that generates responses from collections of known conversations.

---

Install dependencies:
```bash
pip install -r requirements.txt
```

On first run, the script will download the spaCy English model (`en_core_web_sm`) if it is not already installed.


# How to run the program
- From the `Assignment03` directory, run: `python myChatbot.py`
- Type your messages at the `You:` prompt and press Enter.
- Type `quit` or `exit` to end the conversation (or use Ctrl+C).


# Output
- The bot replies at the `Bot:` prompt with responses from the English corpus.
- Example exchange:
  - **You:** Good morning! How are you doing?
  - **Bot:** I am doing very well, thank you for asking.
  - **You:** You're welcome.
  - **Bot:** Do you like hats?
- A SQLite database (`db.sqlite3`) is created in the same directory to store conversation data.


## Requirements
- Python 3.9+ (ChatterBot supports Python &lt;3.14)
- `django`
- `chatterbot`
- `chatterbot-corpus`
- `spacy` (and the `en_core_web_sm` model)
- `pyyaml`
