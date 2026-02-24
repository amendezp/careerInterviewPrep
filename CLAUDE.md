# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Purpose

This is a personal workspace for career and financial research. There is no application source code, build system, or test suite.

## Available Tools

- **pdftotext**: Installed via Homebrew at `/opt/homebrew/bin/pdftotext`. Used for extracting text from PDF documents (resumes, financial filings, reports).
- **Web access**: Configured for fetching content from `www.sec.gov`, `www.cnbc.com`, and `www.businessinsider.com`.

## Notes

- Homebrew binaries are at `/opt/homebrew/bin` (Apple Silicon Mac).
- When processing PDFs, use `pdftotext` rather than attempting to read PDF files directly.
