# Regenerate all man pages from argparse parsers and hand-written sources.
# Usage: make -C man   OR   make man (from project root via pyproject.toml scripts)

PYTHON ?= python3
ARGPARSE_MANPAGE ?= argparse-manpage

.PHONY: all clean

all: man1/cogtrix.1 man1/cogtrix-api.1

man1/cogtrix.1: ../cogtrix_core/cli/args.py
	$(ARGPARSE_MANPAGE) \
	  --pyfile ../cogtrix_core/cli/args.py \
	  --function build_parser \
	  --project-name cogtrix \
	  --prog cogtrix \
	  --author "Northland Positronics" \
	  --url "https://github.com/NorthlandPositronics/Cogtrix" \
	  --manual-title "Cogtrix Manual" \
	  --description "AI-powered conversational agent and tool-execution platform" \
	  > man1/cogtrix.1

man1/cogtrix-api.1: ../cogtrix_core/api/__main__.py
	$(ARGPARSE_MANPAGE) \
	  --pyfile ../cogtrix_core/api/__main__.py \
	  --function build_parser \
	  --project-name cogtrix \
	  --prog cogtrix-api \
	  --author "Northland Positronics" \
	  --url "https://github.com/NorthlandPositronics/Cogtrix" \
	  --manual-title "Cogtrix Manual" \
	  --description "Cogtrix REST API server" \
	  > man1/cogtrix-api.1

clean:
	rm -f man1/cogtrix.1 man1/cogtrix-api.1
