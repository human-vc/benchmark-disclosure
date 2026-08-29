.PHONY: test numbers panel check

test:
	python -m pytest tests/ -q

panel:
	python -m src.build_matrix

numbers:
	python -m src.paper_numbers

check: test
	python -m src.snapshot
	python -m src.paper_numbers

paper:
	./paper/build.sh 9

anonymous:
	./paper/anonymize.sh

overleaf:
	python3 paper/flatten.py
