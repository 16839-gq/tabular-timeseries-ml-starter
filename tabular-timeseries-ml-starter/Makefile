.PHONY: install demo-data demo test lint format clean

install:
	python -m pip install -e '.[dev]'

demo-data:
	python scripts/make_demo_data.py

demo: demo-data
	python -m ds_starter.cli run --config configs/tabular_demo.yml
	python -m ds_starter.cli run --config configs/timeseries_demo.yml

test:
	python -m pytest -q

lint:
	ruff check .
	
format:
	ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache build dist
