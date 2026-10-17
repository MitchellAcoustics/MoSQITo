<div align="center">
    <img src="https://raw.githubusercontent.com/Eomys/MoSQITo/master/logo.png">
</div>

## What is MOSQITO ?

The objective of MOSQITO is to provide a unified and modular development framework of key sound quality tools (including key SQ metrics), favoring reproducible science and efficient shared scripting among engineers, teachers and researchers community. The development roadmap of the project is presented in more details in the Scope of the project.

It is written in Python, one of the most popular free programming language in the scientific computing community. It is meant to be highly documented and validated with reference sound samples and scientific publications.

## Installation

```
pip install mosqito
```

Optionally, `pip install "mosqito[performance]"` installs [numba](https://numba.pydata.org), which compiles the loops of the time-varying Zwicker loudness for a large speed-up. Totals and specific loudness are unchanged to within 1e-12 (tested against the ISO 532-1 reference signals).

## Documentation

Full documentation available on the [ReadTheDocs website](https://mosqito.readthedocs.io/en/latest/) !

