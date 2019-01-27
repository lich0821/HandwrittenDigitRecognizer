#! /usr/bin/env python
# -*- coding: utf-8 -*-

from app import create_app
from pathlib import Path

app = create_app()

if __name__ == "__main__":
    if not Path('./.uploads').exists():
        Path('./.uploads/').mkdir()

    app.debug = True
    app.run()
