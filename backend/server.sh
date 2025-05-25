#!/usr/bin/sh

uvicorn main:app --reload --reload-exclude="*.log"