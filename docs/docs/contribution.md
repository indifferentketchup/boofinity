# Contribute and Develop

## Developer setup

Install via Poetry 1.8.4 and Python3.11 on Ubuntu 22.04
```bash
git clone https://github.com/michaelfeil/infinity
cd infinity
cd libs/boofinity
poetry install --extras all --with test
```

To ensure your contributions pass the Continuous Integration (CI), there are some useful local actions.
The `libs/boofinity/Makefile` is a useful entrypoint for this.
```bash
cd libs/boofinity
make format
make lint
make template-docker
poetry run pytest ./tests
```

As an alternative, you can also use the following command, which bundles a range of the above.
```bash
cd libs/boofinity
make precommit
```

## CLA
Infinity is developed as open source project. 
All contributions must be made in a way to be compatible with the MIT License of this repo. 