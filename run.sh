#!/bin/bash

echo " Menjalankan pipeline Data Mining..."

if [ -f requirements.txt ]; then
    echo " Menginstall dependensi dari requirements.txt..."
    pip install -r requirements.txt
fi

echo " Menjalankan src/main.py..."
python3 src/main.py