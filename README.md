# Autocomplete Engine

A fast, case-insensitive autocomplete engine for substring search with fuzzy matching (≤1 typo allowed). Designed for large text corpora, using trigram indexing and efficient filtering.
This project was part of the "Exellenteam in academy" program with collaboration of Google.

## Features
- Substring search (not just prefix)
- Case-insensitive
- Allows up to 1 typo (insertion, deletion, or substitution)
- Fast lookup using trigram index
- Returns scored, ranked results
- Includes timing and performance tests

## Project Structure

- `ac_engine.py` — Core autocomplete engine and CLI
- `config.py` — Paths and configuration
- `json_init.py` — Script to build a JSON dataset from text files in `Archive/`
- `test_AC.py` — Comprehensive unit and performance tests
- `Archive/` — Folder with source text files (input data)
- `sentences.json` — Generated JSON data (output of `json_init.py`)
- `sentences.gz` — Serialized, indexed engine (auto-generated)

## Setup

1. **Install dependencies:**
   - Python 3.10+
   - (Optional) `python-Levenshtein` for faster edit distance
   - `orjson` (used but not required for core functionality)

   ```bash
   pip install orjson python-Levenshtein
   ```

2. **Prepare data:**
   - Place your `.txt` files in the `Archive/` directory.
   - Run the data initialization script:

   ```bash
   python json_init.py
   ```
   This creates `sentences.json` from all text files in `Archive/`.

3. **Build the index:**
   - The first run of the engine will automatically build and save the index as `sentences.gz`.

## Usage

### Command Line

Run the engine interactively:

```bash
python ac_engine.py
```

- Enter a query to get autocomplete suggestions.
- Results show the best matches, their scores, and source file/line.
- Use `#` at the end of a query to reset.

### As a Python Module

You can import and use the engine in your own code:

```python
from ac_engine import load_engine
engine = load_engine()
results = engine.get_best_k_completions('your query', allow_one_typo=True, topn=5)
for r in results:
    print(r.completed_sentence, r.score, r.source)
```

## Testing

Run all tests (including performance and edge cases):

```bash
python -m unittest test_AC.py
```

## Notes
- The engine uses a trigram index for fast candidate filtering.
- Fuzzy matching is limited to a single edit (insertion, deletion, or substitution).
- The `sentences.gz` index is auto-generated and can be deleted to force a rebuild.
