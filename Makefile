.PHONY: test numbers panel check

test:
	python -m pytest tests/ -q

panel:
	python -m src.build_matrix

# Regenerates every number the manuscript cites into data/paper_numbers.json,
# stamped with the data snapshot. Nothing is typed into the paper by hand.
numbers:
	python -m src.paper_numbers

# What to run before claiming a number is true.
check: test
	python -m src.snapshot
	python -m src.paper_numbers

# Compile the manuscript and report errors, undefined references, overfull
# boxes and the page count. Run after every edit, never batched.
paper:
	./paper/build.sh 9

# Anonymous supplementary archive for double-blind submission. Builds a fresh
# tree from tracked files with no version-control history, since the identity
# leak is entirely in git metadata rather than in the source.
anonymous:
	./paper/anonymize.sh
