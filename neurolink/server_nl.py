#!/usr/bin/env python3
"""Lance le serveur NeuroLink (API style Ollama).  Ex. : python server_nl.py --port 11434"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neurolite.server import main  # noqa: E402

if __name__ == "__main__":
    main()
