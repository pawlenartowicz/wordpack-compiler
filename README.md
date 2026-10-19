# wordpack-compiler

Builds **wordpacks** for [SSD (Supervised Semantic Differential)](https://github.com/hplisiecki/Supervised-Semantic-Differential).
A wordpack is one file per embedding and language. It holds the word vectors,
the vocabulary, the stopwords and a self-test, so SSD can turn text into
vectors without spaCy or any other NLP tool at run time.

All language processing happens here, at build time. Readers only tokenize and
look words up.

A wordpack does not depend on SSD. Any program that needs word vectors for
text can read one, for example to find words similar to a word, or sentences
and text fragments similar to a query.

## Available packs

| tag | full name | language | use it for | trained on |
|---|---|---|---|---|
| `en`, `en-dolma`, `default` | `en-glove2024-dolma-1` | English | informal or recent text: comments, social media, open survey answers, forums | Dolma v1.6: web pages, Reddit to 2023, books |
| `en-wiki` | `en-glove2024-wikigiga-1` | English | formal text: news, essays, reports, older press | Wikipedia 2024 and newswire 1994–2010 |

Both are GloVe 2024, 300 dimensions, lowercase, licence PDDL-1.0. Dolma has
1.2M words, Wikigiga 1.29M. Source:
<https://downloads.cs.stanford.edu/nlp/data/wordvecs/>. Citation: Carlson,
Bauer & Manning (2025). A New Pair of GloVes. arXiv:2507.18103.

Load a pack by tag or by full name:

```python
from ssdiff import Embeddings

emb = Embeddings.load("en")                    # tag: the English default of the installed ssdiff
emb = Embeddings.load("en-glove2024-dolma-1")  # full name: always this file
```

A full name always means the same file. A tag can be moved to a newer pack
by a later ssdiff version, so use the full name where results must be
reproducible.

## Build a pack

```bash
pip install -e .
wordpack-compiler build en/glove2024-dolma
wordpack-compiler info build/en-glove2024-dolma-1.wordpack
```

`build` downloads the source embedding into `build/sources/` and writes the
pack to `build/<name>-<version>.wordpack`. The version is a whole number: one
more than the highest version of that pack in `index.json`, or 1 when it has
none. `info` prints what the pack contains and runs its self-test.

## Publish a pack

Publishing needs write access to the bucket. Put the settings in
`.publish.toml` in the repository root (git-ignored):

```toml
endpoint = "https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
bucket = "..."
access_key_id = "..."
secret_access_key = "..."
```

Then:

```bash
pip install -e .[publish]
wordpack-compiler build en/glove2024-dolma
wordpack-compiler info build/en-glove2024-dolma-1.wordpack
wordpack-compiler publish build/en-glove2024-dolma-1.wordpack
```

`publish` runs the pack's self-test, uploads the pack, adds its version to
`index.json`, and uploads `index.json`. It stops before uploading anything
if the bucket's `index.json` lists a version the local one does not. A published
file is never replaced: changed contents need a new build, which gets the
next version. Commit `index.json` after publishing; its history is the
record of what was published.

Publishing a pack does not move any tag. The tags are listed in ssdiff, in
`ssdiff/lang_config.py`: to add or move one, edit `WORDPACK_TAGS` there and
release ssdiff.

## Add a language

A language is one folder under `languages/`, named by its ISO 639-1 code.
Copy `languages/en/` and edit:

1. **`language.toml`**
   - `keys`: `"surface"`, meaning the embedding stores words as they appear
     in text.
   - `case_sensitive`: whether the embedding distinguishes upper and lower case.
   - `clitics`: word endings the embedding's tokenizer split off, such as
     English `n't` and `'s`. Check the embedding's vocabulary to find them.
   - `neighbor_filter`: a regex for keys to hide when SSD lists nearest words.
   - `stopwords_source`: where `stopwords.txt` comes from, with its licence.
2. **`stopwords.txt`**: one word per line.
3. **`selftest.txt`**: about 20 sentences in the language, one per line. Use
   contractions, capitals, digits, hyphens and rare words, because the
   self-test is how different readers are checked against each other.
4. **One recipe per source embedding**, e.g. `glove2024-dolma.toml`: pack
   name, description, download URL, licence and citation.

Then run `build` and `info` on your pack.

For now, a language can be added without code changes if:

- its embedding stores surface forms, not lemmas (Polish NKJP embeddings, for
  example, store lemmas and need a form table the compiler does not build yet);
- words are separated by spaces (not Chinese, Japanese or Thai);
- its words contain no combining marks (Unicode categories Mn and Mc), since
  the tokenizer ends a word at one (not Hindi or other Indic scripts);
- the embedding's licence allows redistribution, since the pack contains its
  vectors.

## Format

A wordpack is a valid [safetensors](https://github.com/huggingface/safetensors)
file. [`FORMAT.md`](FORMAT.md) defines its contents. A reader in another
language follows `FORMAT.md` and must pass the self-test stored in every pack.
`wordpack_compiler/reader.py` is the reference reader.

## Licence

The code is licensed GPL-3.0-or-later. Each pack carries the licence of its
source embedding (GloVe: PDDL). `testpack/data/warriner2013/` has its own
licence, CC BY-NC-ND 3.0. See the README in that folder.

## Citation

If you use SSD, cite: Plisiecki, H., Lenartowicz, P., Pokropek, A., Małyska,
K., & Flakus, M. (2025). Measuring Individual Differences in Meaning: The
Supervised Semantic Differential. PsyArXiv.
<https://doi.org/10.31234/osf.io/gvrsb_v1>

---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
