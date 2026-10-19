# The .wordpack format

Format version 1. The reference reader is wordpack_compiler/reader.py.

## What a pack holds

A pack holds all the data a reader needs to go from text to vectors:

- the rows;
- the ABTT mean and components;
- the vocabulary and its lookup table;
- the stopwords;
- the masked-word list;
- the character table: which code points are letters, digits and
  whitespace, how each one lowercases and uppercases, and which are folded
  before tokenizing;
- the tokenizer settings;
- the neighbour filter;
- the self-test.

The tokenizer rules themselves are code in the reader. The format version
fixes them, and the self-test checks them.

- **Rules are code.** They are not stored as regexes. Python's `re` and R's
  ICU read the same pattern differently, so a stored regex would not make the
  rules portable.
- **Character data comes from the pack.** A reader takes its character
  classes and case maps from the pack's character table, never from its own
  runtime. Python 3.10, Python 3.14 and R each ship a different Unicode
  version. With the table, they still tokenize the same text the same way,
  and the self-test can pin the result.

On the wordpack route `ssdiff` reads nothing but the pack. It uses no bundled
stopword list, no `lang_config` and no spaCy.

## File

A wordpack is a valid safetensors file:

- an 8-byte little-endian header length;
- a JSON header, padded with spaces to a multiple of 8 bytes;
- the tensor bytes, back to back, with no gaps.

The compiler writes the F32 and I32 tensors first (`abtt`, `norms`,
`vocab_order`), so they start at 4-byte aligned offsets. A reader takes each tensor's position from the header, never from
this order.

A reader identifies a wordpack by the `format` metadata key, not by the
extension.

## Tensors

| tensor | dtype | shape | content |
|---|---|---|---|
| `abtt` | F32 | (11, D) | row 0 is the mean, rows 1–10 the top 10 principal components by eigenvalue; see "Rows" |
| `norms` | F32 | (N,) | length of each source row before L2 |
| `vocab_order` | I32 | (N,) | row ids sorted by the UTF-8 bytes of their keys; see "Lookup" |
| `matrix` | I8 | (N, D) | one row per key, in source order (frequency order) |
| `vocab` | U8 | (bytes,) | UTF-8, one key per line, `\n` after each; line index = row id from 0 |
| `stopwords` | U8 | (bytes,) | UTF-8, lowercase, one word per line, `\n` after each |
| `masked` | U8 | (bytes,) | UTF-8, lowercase, one word per line, `\n` after each; may be empty; see "Tokenizing", step 8 |
| `chars` | U8 | (bytes,) | UTF-8 JSON, the character table; see "Characters" |
| `selftest` | U8 | (bytes,) | UTF-8 JSON; see "Self-test" |

A later format version may add tensors. A reader ignores tensors it does not
know.

## Metadata

safetensors allows only string values in `__metadata__`, so every value is a
string. Lists and objects are JSON strings.

| key | example | from |
|---|---|---|
| `format` | `wordpack` | fixed |
| `format_version` | `1` | fixed |
| `name` | `en-glove2024-dolma` | recipe |
| `version` | `1` | `index.json` |
| `description` | `Informal or recent text: …` | recipe |
| `source` | `{"url": …, "license": "PDDL-1.0", "citation": …}` | recipe |
| `language` | `en` | `language.toml` `code` |
| `keys` | `surface` | `language.toml` |
| `case_sensitive` | `false` | `language.toml` |
| `clitics` | `["n't", "'s", …]` | `language.toml` |
| `neighbor_filter` | regex | `language.toml` |
| `stopwords_source` | `spaCy 3.8.13 English stop words, MIT licence` | `language.toml` |
| `quantization` | `int8-rowmax` | fixed |
| `compiler_version` | `0.1.0` | package version |

`name` is made of lowercase letters, digits and hyphens, and does not end in
a hyphen followed only by digits. `version` is a whole number: `build` sets
it to one more than the highest version of that `name` in the repository's
`index.json`, or to 1 when there is none. `<name>-<version>` is the pack's
full name, and its file is `<name>-<version>.wordpack`.

A reader refuses a pack when:

- `format` is not `wordpack`;
- `format_version` is not one it knows;
- `keys` is not `surface`. Lemma-keyed packs (Polish, later) add a form table
  that a version 1 reader does not apply, so it would tokenize them wrongly
  without noticing;
- `case_sensitive` or `clitics` is missing;
- the header is not a JSON object, `clitics` or `source` is not valid JSON,
  or a tensor entry lacks a known `dtype`, a `shape` of non-negative
  integers (two of them for `matrix`) or two `data_offsets` with start ≤ end;
- any of the nine tensors in "Tensors" is missing;
- the file ends before a tensor's data does, as after a cut download or
  copy.

It ignores metadata keys it does not know.

## Rows

**Building.** The compiler:

1. L2-normalizes each source row.
2. Quantizes it (`int8-rowmax`): it scales the row so its largest absolute
   component maps to 127 and rounds. -128 is never written. No scale is
   stored: rows are unit-length, so a reader gets the direction back by
   L2-normalizing.
3. Computes `abtt` from the rows as a reader sees them at `m = 0`, that is,
   converted to float and L2-normalized. It takes the mean and the
   eigenvectors of the D × D covariance, in float64 over all N rows. `abtt`
   therefore describes the stored matrix, not the float source.

**Reading with ABTT `m`**, for `m` from 0 to 10, default 1:

1. Convert the int8 row to float32 and L2-normalize it.
2. If `m > 0`:
   - subtract `abtt[0]`;
   - remove the projection on `abtt[1 : m+1]`;
   - L2-normalize again.

No arithmetic is done on int8 values.

The result equals `ssdiff`'s `Embeddings.normalize(l2=True, abtt=m)` run on
the whole `m = 0` matrix, up to float32 error.

## Lookup

The lookup table is part of the file, so a reader builds no dictionary:

- Keys are unique after NFC, fold and, when `case_sensitive` is false,
  lowercasing; the compiler drops later duplicates.
- A reader finds a key's row by binary search over `vocab_order`, comparing
  UTF-8 bytes, not locale-aware strings, so Python and R agree on the order.
- The start of each key in `vocab` comes from the positions of the `\n`
  bytes. A reader computes them at load, in one pass over the bytes.
- The same search answers whether any key starts with a given prefix: find
  where the prefix would sort, then check whether the key there starts with
  it. The vocabulary match in "Tokenizing" uses this.
- A lookup is exact: no case change and no normalization. The case rule
  belongs to tokenizing.

For 1.2M keys, the vocabulary bytes, the key offsets and `vocab_order` take
about 20 MB in RAM. A Python list plus dict of the same keys takes 159 MB.
Both were measured on the first 1.2M keys of GloVe 42B, which is lowercased
like GloVe 2024 (10.7 MB of key bytes, 7.9 bytes per key).

## Characters

The `chars` tensor is a JSON object, written by the compiler from its own
Python's Unicode data:

```json
{
  "unicode_version": "15.0.0",
  "letter": [[65, 90], [97, 122], …],
  "digit": [[48, 57], …],
  "space": [[9, 13], [28, 32], …],
  "lower": {"65": "a", "304": "i\u0307", …},
  "upper": {"97": "A", …},
  "fold": {"173": "", "8203": " ", "8217": "'", …}
}
```

- **`letter`, `digit`, `space`.** Sorted, inclusive code-point ranges.
  - A letter is general category L, Nl or No. These are the code points
    Python's `[^\W\d_]` matches.
  - A digit is general category Nd.
  - Whitespace is what Python's `str.isspace` accepts.
  - A reader uses these ranges for every "letter", "digit" and "whitespace"
    in "Tokenizing", and for nothing else.
- **`lower`, `upper`.** For each code point whose full lowercase or uppercase
  mapping differs from itself, the replacement string, which may be longer
  than one code point (`İ` lowercases to `i` + U+0307).
  - Lowercasing or uppercasing a string replaces each code point on its own.
  - There are no context rules: `ΟΣ` lowercases to `οσ`, not to `ος` as
    Python's `str.lower` would give.
- **`fold`.** Replacements applied to the whole text after NFC (Tokenizing,
  step 1):
  - every format character (general category Cf) is deleted, except
    U+200B. This covers the soft hyphen, the zero-width joiner and non-joiner,
    the word joiner, the byte order mark and the bidirectional controls;
  - U+200B ZERO WIDTH SPACE becomes a space;
  - `’`, `‘` and `ʼ` (U+2019, U+2018, U+02BC) become `'`;
  - U+2010 HYPHEN and U+2011 NON-BREAKING HYPHEN become `-`.

  Invisible characters inside words show up in real data: soft hyphens
  between letters in the hate-speech comments, and zero-width joiners and
  invisible separators in Kalibra.
- **`unicode_version`.** The Unicode version of the compiler's Python, for
  reference only.

NFC is the one step that still uses the reader's own Unicode data. Unicode
guarantees NFC gives the same result in every version for characters that
existed in the older one. So readers can differ only on characters assigned
after their own version, and only when such a character has a decomposition.

## Tokenizing

`tokenize(text)` turns a text into sentences, keys and rows. The rules:

1. **NFC, fold, emphasis stars.** The text is normalized to NFC, then the
   `fold` map from "Characters" is applied, then paired emphasis stars are
   replaced by spaces. Every offset the reader returns indexes this text,
   counted in Unicode code points, not bytes, so Python and R agree. For text
   in NFC with no folded characters and no paired stars, which is almost all
   text, it is the input unchanged.
   - **Emphasis stars.** Markdown marks emphasis with stars, as in `*word*`
     and `**two words**`. Left in, the closing stars would join the masked
     run of step 8, and `**bit**` would be read as a five-letter masked word.
   - A star run is a maximal run of `*` outside the spans that step 3 skips.
     The reader takes the star runs of each line from left to right. Lines
     end at U+000A. For each run length it remembers at most one open run.
     - A run closes when an open run of the same length is remembered, the
       character before the run is not whitespace, and the character after
       it is not a letter or a digit. Every star of both runs becomes a
       space, and the open run is forgotten.
     - Otherwise a run opens when the character after it is not whitespace
       and the character before it is not a letter or a digit. It is
       remembered in place of any earlier open run of its length.
     - The start and the end of the text count as whitespace.
     - At a line's end every open run is forgotten.
   - Examples:
     - `a **bit** more` becomes `a   bit   more`;
     - `*f**k*` becomes ` f**k `, which step 8 reads as `fuck`;
     - `sh*t`, `f***`, `*never` and `5*x` are unchanged: none of them has
       both an opening and a closing run;
     - `https://*.example.org/*` is unchanged: both stars are inside a
       skipped span.
   - This reads wrongly a masked word whose closing stars match an open run
     on its line: `**this sh** is real` gives `sh`, not `shit`. Emphasis that
     crosses a line feed is not paired.
2. **Sentences.**
   - A sentence ends after a run of `.`, `!` or `?` that is followed by
     whitespace or by the end of the text, and after every line feed
     (U+000A). No other character counts as a line break.
   - A sentence's span runs from its first non-whitespace character to its
     last one.
   - Pieces that hold only whitespace are not sentences.
   - This splits wrongly after abbreviations such as `Dr.`, and does not
     split at `."` followed by a space.
3. **Skipped spans.** Inside a sentence, these produce no tokens:
   - `https?://\S+`;
   - `@\S+` at the start of the text or after whitespace.

   These are the two patterns `ssdiff` drops on the spaCy route today
   (`_URL`, `_AT` in `utils/text.py`).
4. **Characters.** Letter, digit and whitespace mean the pack's `letter`,
   `digit` and `space` ranges ("Characters"). `\S` in step 3 is any code point
   outside `space`. Lowercasing and uppercasing use the pack's `lower` and
   `upper` maps, one code point at a time.
5. **Case rule.** Resolving a form gives a key and a row:
   - When `case_sensitive` is false, the key is the form lowercased, and it
     is looked up once.
   - When `case_sensitive` is true, the reader tries up to three case
     candidates in order, and the first with a row wins:
     1. the form as written;
     2. the form lowercased;
     3. the form with its first code point uppercased (the `upper` map, not
        titlecase) and the rest lowercased.

     So `polish` finds `polish`, `Polish` finds `Polish`, and `POLSKA` and
     `polska` both find `Polska` in a vocabulary that has only `Polska`.
   - When no case candidate has a row, the key is the lowercased form and the
     row is -1.
6. **Chunks and units.**
   - Inside a sentence and outside skipped spans, a chunk is a maximal run of
     non-whitespace characters.
   - A chunk is made of units: a run of letters, a run of digits, or any
     single other character.
   - The reader walks each chunk from left to right. At each unit that starts
     with a letter or a digit, it tries steps 7 to 10 in order.
   - **Stray clitics.** At a `'`, the reader checks for a clitic first.
     - If a clitic starting with `'` begins there (compared lowercased) and is
       followed by the end of the chunk or by a non-letter, that clitic
       becomes one token. When several qualify, the first in list order
       wins.
     - This catches a clitic left after a vocabulary match, a masked word or
       a digit run:
       - `90's` gives `'s`;
       - `covid-19's` gives `covid-19` and `'s`;
       - `sh*t's` gives `shit` and `'s`.

       Without this check the `'` would be skipped and `s` would become a
       word of its own.
     - `'so'` gives `so`, because a letter follows `'s`.
   - The reader skips any other character.
7. **Vocabulary match.**
   - Starting at this unit, the reader grows a candidate one unit at a time.
     It keeps growing while some key starts with the candidate, in any of its
     case candidates. The reader answers that with one binary search over
     `vocab_order`.
   - Of the candidates seen, the longest one that meets all of these becomes
     one token:
     - it has a row under the case rule;
     - it contains a letter;
     - it contains a digit, or a character other than a letter, `-` and `'`;
     - it ends in a letter or a digit, or it ends in `.` and has another `.`
       before that.
   - The walk then continues after the token.
   - Examples:
     - `covid-19`, `4chan`, `3rd`, `ms-13`, `and/or`, `u.s.` and `e.g.` each
       become one token when they are keys;
     - `e.g.,` gives `e.g.`.
   - The last condition keeps junk keys out. GloVe 42B has `it.` at rank
     22,257 and `you.` at rank 36,650, and a sentence-final `it.` must give
     `it`.
   - A chunk made only of letters, `-` and `'` can never produce a match, so
     a reader may skip this step for such a chunk. The output is the same.
8. **Masked words.** If there was no vocabulary match and the unit is
   letters, the reader looks for a masked word:
   - The masked run is the longest run from this unit made of letters and
     `*`, holding at least one `*`.
   - Each `*` stands for exactly one letter.
   - A word in `masked` matches when it has the run's length and the same
     letter, compared lowercased, at every position without a `*`.
   - If exactly one word matches, the run becomes one token. Its form is that
     word, resolved with the case rule. If none matches, or more than one,
     the reader goes on to step 10 and the stars are skipped as other
     characters.
   - Examples, with `fuck` and `shit` in the list:
     - `sh*t` gives `shit`, and `f**k` gives `fuck`;
     - `f***` gives `fuck`;
     - `*never` gives `never`. The run starts at a letter, so a leading
       star is never part of it.
   - The vocabulary match runs first, so a pack whose source has `f*ck` as a
     key uses that key's own vector.
9. **Digits.** If there was no match and the unit is digits, the unit is
   skipped. A number alone is never a token, as on the spaCy route.
10. **Word.** If there was no match and the unit is letters, the word is the
   longest match of `L+(['-]L+)*` starting here, where `L` is a letter as
   defined in step 4. The walk continues after it.
   - **Clitics.** If the lowercased word ends with one of `clitics` and
     something is left before it, the word splits into two tokens: the stem
     and the clitic. The first match in list order wins, and at most one
     clitic is split off per word. `don't` becomes `do` + `n't`, and `can't`
     becomes `ca` + `n't`, as CoreNLP split them for GloVe 2024.
   - **Clitics come before any lookup**, so `don't` never resolves to the key
     `don't`. GloVe 42B has it, at rank 505,337.
   - **Hyphens.** A stem containing `-` is resolved whole. If it has no row,
     each part between hyphens is treated as a word of its own: the clitic
     rule, then resolving. So `don't-care` gives `do`, `n't` and `care`.
     An empty part gives no token: `dog-n't` splits into `dog-` and `n't`,
     and `dog-` gives only `dog`.
   - **Resolving.** Each resulting form is resolved with the case rule.
   - **Examples.** When `covid-19` is not a key, it yields `covid`. When `b2b`
     is not a key, it yields `b` and `b`.

Every token has its own span in the text:

- a vocabulary match's span is the candidate;
- a masked word's span is the masked run;
- a clitic's span is the last characters of the word, or the stray clitic
  itself;
- a hyphen part's span is that part.

A token is a stopword when its lowercased key is in `stopwords`. Tokenizing
does not remove stopwords, and it does not drop tokens without a row.
`ssdiff` filters both itself, so `stopwords=` can override the pack's list.

The rules are fixed by the format version. A pack cannot change them, apart
from `case_sensitive`, `clitics` and the `masked` list.

## Self-test

`selftest` is a JSON object:

```json
{
  "texts": [
    {"text": "…",
     "sents": [[0, 41]],
     "keys": ["i", "do", "n't", …],
     "spans": [[0, 1], [2, 4], [4, 7], …],
     "sent": [0, 0, 0, …],
     "rows": [12, 30, 47, …]}
  ],
  "vectors": [
    {"row": 0, "abtt": 0, "values": […]},
    …
  ]
}
```

- **`texts`**: one entry per line of the language's `selftest.txt`, then the
  four fixed texts below, with -1 in `rows` where there is no row. Lines are
  split at U+000A only, and lines holding only whitespace are skipped. A line
  may hold several sentences.
- **Fixed texts.** Every pack carries these four, shown as JSON strings. The
  line-break rules are the same in every language, and a line of
  `selftest.txt` cannot hold a line feed:
  - `"one\ntwo"`: a line feed ends a sentence;
  - `"one\r\ntwo"`: the `\r` is not part of the first sentence;
  - `"one.\n\n \t\ntwo"`: pieces holding only whitespace are not sentences;
  - `"one\rtwo\u000bthree\u000cfour\u0085five\u2028six\u2029seven\u001cend"`:
    no other character counts as a line break.
- **`vectors`**: rows 0 and N−1, each at `abtt` 0, 1 and 3, as float32 values
  written in full precision. Row N−1 checks offset arithmetic at the end of
  the file. `abtt` 1 and 3 check the single-component and multi-component paths.

A reader passes when:

- token output matches exactly;
- every vector value is within 1e-5 of the stored one, which leaves room for
  float differences between BLAS libraries and between Python and R.

On failure it raises, naming the pack, the reader's format version and the
first mismatch.
