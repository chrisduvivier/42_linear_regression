
all:
	python3 -m pip install --user virtualenv
	python3 -m venv .venv
	( \
       source .venv/bin/activate; \
       python3 -m pip install -r ./requirements.txt; \
    )

clean:
	rm -rf .venv