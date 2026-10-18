# P3 pinned literary corpus

Original UTF-8 files from Project Gutenberg are retained under `raw/`, including their full source notices and license.
Catalogs mark the works “Public domain in the USA”; the original terms distinguish the underlying text from
Project Gutenberg branding. See [source license](https://www.gutenberg.org/policy/license.html) and the
catalog URLs, authors, retrieval date and exact SHA-256 in `sources.json`. No worldwide rights conclusion is implied.

Training excludes source headers, contents and license boilerplate. Chapters are documents; exact duplicate
chapters and repeated paragraphs of at least80 normalized characters are removed before deterministic book-stratified
splitting. A cross-split word5-gram Jaccard audit detects near-identical chapters. This does not prove absence of
shared motifs, short phrases or every form of contamination. The test is held-out chapters within the same three books,
not a held-out author/book generalization benchmark. BPE learns only from train chapters.

Snapshots are bundled so training does not download data. Source changes fail the hash check rather than silently
changing the experiment. Use `kaleido.lm.corpus.prepare_corpus` to regenerate the document audit.
