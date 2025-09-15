#!/usr/bin/env bash

echo "Running bitstream loading check..."
sudo -E /usr/local/share/pynq-venv/bin/python /cg4002/AI/test_bitstream_loading.py
echo "Done."