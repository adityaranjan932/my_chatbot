#!/bin/bash
gunicorn main:app -c gunicorn_config.py
