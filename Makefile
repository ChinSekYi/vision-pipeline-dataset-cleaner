install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	isort .
	black .

run:
	python main.py --input data/original_raw --output data/final

evaluate:
	python evaluate.py

run-help:
	python main.py --help