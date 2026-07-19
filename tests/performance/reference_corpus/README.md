# Reference performance corpus

This directory contains the 359 preprocessed dialogue transcripts that shipped
with the repository's initial commit (`109e004`, 2022-09-29). The original test
fixture described the collection as a combination of:

- Du Bois, John W., Wallace L. Chafe, Charles Meyers, Sandra A. Thompson, Nii
  Martey, and Robert Englebretson (2005). *Santa Barbara Corpus of Spoken
  American English*. Philadelphia: Linguistic Data Consortium.
- Newman, John and Georgie Columbus (2010). *The International Corpus of English
  – Canada*. Edmonton, Alberta: University of Alberta.

The fixture also recorded that the transcripts were lowercased and that
punctuation was replaced by alphanumeric substitutions (for example, `_`
became `undrscr`). Each line generally represents a dialogue turn. No more
detailed preprocessing recipe is present in the repository history.

These files are retained as a realistic, stable performance workload. They are
not labeled evaluation data and their outputs are not a correctness oracle.

The repository does not currently record the upstream download locations,
license terms, source-file checksums, or checksums for this processed snapshot.
Verify and document those details against the upstream corpora before
redistributing or treating this copy as authoritative; do not infer licensing
from the citations above.
