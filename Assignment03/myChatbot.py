import os
import sys

# Configure Django before django.setup() so ChatterBot can use Django storage
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY="terminal-chatbot-dev-key-change-in-production",
        # SQLite database in same directory as this script
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": os.path.join(os.path.dirname(os.path.abspath(__file__)), "db.sqlite3"),
            }
        },
        # Django apps required for ChatterBot's DjangoStorageAdapter
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "chatterbot.ext.django_chatterbot",
        ],
        # ChatterBot: store statements in Django DB, use BestMatch for replies
        CHATTERBOT={
            "name": "Terminal Chatbot",
            "storage_adapter": "chatterbot.storage.DjangoStorageAdapter",
            "logic_adapters": ["chatterbot.logic.BestMatch"],
        },
    )

django.setup()  # Initialize ORM so migrations and ChatterBot can run

from django.core.management import call_command
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer


def ensure_spacy_model():
    """Download the spaCy English model if not already installed (required by ChatterBot)."""
    try:
        import spacy
        spacy.load("en_core_web_sm")  # Throws OSError if model missing
    except OSError:
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
            capture_output=True,
        )


def run_migrations():
    """Run Django migrations for the ChatterBot app."""
    call_command("migrate", "django_chatterbot", verbosity=1)


def main():
    ensure_spacy_model()
    run_migrations()

    # Create bot from Django settings and train on English corpus
    chatbot = ChatBot(**settings.CHATTERBOT)
    trainer = ChatterBotCorpusTrainer(chatbot)
    trainer.train("chatterbot.corpus.english")

    print("Terminal Chatbot ready. Type 'quit' or 'exit' to end the conversation.\n")

    # Conversation loop: read input, get bot response, print until user quits
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            response = chatbot.get_response(user_input)
            print(f"Bot: {response}\n")
        except KeyboardInterrupt:  # Ctrl+C
            print("\nGoodbye!")
            sys.exit(0)
        except EOFError:  # Ctrl+D or piped input end
            print("\nGoodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
