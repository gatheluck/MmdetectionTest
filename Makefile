TEST-WORKERS=2

.PHONY: black
black:
	black --check src tests

.PHONY: black-lint
black-lint:
	black src tests

.PHONY: flake8
flake8:
	flake8 src tests

.PHONY: isort
isort:
	isort --check-only src tests

.PHONY: isort-lint
isort-lint:
	isort src tests

.PHONY: mypy
mypy:
	mypy src

.PHONY: test
test:
	pytest tests --cov=src --cov-report term-missing --durations 5

.PHONY: lint
lint:
	$(MAKE) black-lint
	$(MAKE) isort-lint

.PHONY: test-all
test-all:
	$(MAKE) black
	$(MAKE) flake8
	$(MAKE) isort
	$(MAKE) mypy
	$(MAKE) test