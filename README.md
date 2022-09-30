# REMERGE - Multi-Word Expression discovery algorithm

This is a Multi-Word Expression discovery algorithm, which started as a re-implementation and simplification of a multi-word expression (MWE) discovery algorithm called MERGE, detailed in a publication and PhD thesis[^2][^3]. The primary benefit of this algorithm is that it's non-parametric in regards to the size of the n-grams that constitute a MWE—you do not need to specifiy a priori how many n-grams comprise a MWE—you only need to specify the number of iterations you want the algorithm to run.

The code was originally derived from an existing implementation from the original author[^1] that I reviewed, and converted from python 2 to 3, and modified and updated with the following:
- a correction of the log-likelihood calculation; previously it was not using the correct values for the contingency table
- the removal of gapsize / discontinuous bigrams (see below for issues with the prior implementation)
- an overall reduction in codebase size and complexity
  - ~60% reduction in loc
  - removed `pandas` and `nltk` dependencies
- type annotations
- the inclusion of additional metrics (Frequency, [NPMI](https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf)[^4]) for selecting the winning bigram.
- corrections for merging sequetial bigrams greedily and completely.
  - e.g. `'ya ya ya ya'` -> `'(ya ya) (ya ya)'` -> `'(ya ya ya ya)'`. Previously the merge order was non-deterministic, and you could end up with `'ya (ya ya) ya'`
- An overall simplification of the algorithm. 
  - As a tradeoff, this version may be less efficient. After a bigram is merged into a single lexeme in the original implementation, new bigrams and conflicting (old) bigrams were respectively added and subtracted from a mutable counter of bigrams. The counts of this object were difficult to track and validate, and resulted in errors in certain cases, so I removed this step. Instead, only the lexeme data is updated with the new merged lexemes, and bigram data is recreated from scratch using the lexeme data. There is certainly improvement for optimization here, particularly in the case where we are recalculating bigrams from lines that did not contain a merged lexeme and thus their bigram data did not change.

#### Usage

```python
import remerge

corpus = [
    ["a", "list", "of", "already", "tokenized", "texts"],
    ["where", "each", "item", "is", "a", "list", "of", "tokens"],
    ["isn't", "a", "list", "nice"]
]

winners = remerge.run(corpus, iterations=1, method=remerge.SelectionMethod.frequency)

# winners[0].merged_lexeme.word == ('a', 'list')
```

There are 3 bigram winner selection methods: [Log-Likelihood (G²)](https://aclanthology.org/J93-1003.pdf)[^5], [NPMI](https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf)[^4], and raw frequency. They are available under the `SelectionMethod` enum. The default is log-likelihood, which was used in the original implementation.

If using `NPMI` (`SelectionMethod.npmi`), you likely want to provide a `min_count` parameter, "as infrequent word pairs tend to dominate the top of bigramme lists that are ranked after PMI". (p. 4[^4])

```python
winners = remerge.run(corpus, 100, method=remerge.SelectionMethod.npmi, min_count=25)
```

#### Install

Latest release:

```bash
pip install -U remerge
```

For latest from github:

```bash
pip install git+https://github.com/pmbaumgartner/remerge.git 
```

#### Limitations

**No tie-breaking logic** - I found this while testing and comparing to the original reference. If two bigrams are tied for log-likelihood, there is no tie-breaking mechanism. Both this implementation and the original implementation simply pick the first bigram from the index with the maximum value. However, we have slightly different implementations of how the statistics table is created (i.e., the ordering of the index), which makes direct comparisons between implementations difficult.

#### Issues with Original Algorithm

##### Single Bigrams with discontinuities forming from distinct Lexeme positions

One issue with discontinuities / gaps in the original algorithm is that it did not distinguish the position of a satellite lexeme occuring to the left or right of a bigram with a gap.

Take for example these two example sentences, using `-` to represent an arbitrary token:

```
a b c -
a - c b
```

Assume in a prior iteration, a winning bigram was `(a, _, c)`, representing the token `a`, a gap of `1`, and then the token `c`. with a gapsize of 1. The past algorithm would count the token `b` twice towards the same n-gram `(a, b, c)`, despite there being two distinct n-grams represented here: `(a, b, c)` and `(a, _, c, b)`.

I think the algorithm is counting on the fact that it would be very rare to encounter this sequence of lexemes in a realistic corpus, where the same word would appear within the gap **and** after the gap. I think this is more of an artifact of this specific example with an unrealistically small vocabulary.

#### References

[^1]: awahl1, MERGE. 2017. Accessed: Jul. 11, 2022. [Online]. Available: https://github.com/awahl1/MERGE

[^2]: A. Wahl and S. Th. Gries, “Multi-word Expressions: A Novel Computational Approach to Their Bottom-Up Statistical Extraction,” in Lexical Collocation Analysis, P. Cantos-Gómez and M. Almela-Sánchez, Eds. Cham: Springer International Publishing, 2018, pp. 85–109. doi: 10.1007/978-3-319-92582-0_5.

[^3]: A. Wahl, “The Distributional Learning of Multi-Word Expressions: A Computational Approach,” p. 190.

[^4]: G. Bouma, “Normalized (Pointwise) Mutual Information in Collocation Extraction,” p. 11.

[^5]: T. Dunning, “Accurate Methods for the Statistics of Surprise and Coincidence,” Computational Linguistics, vol. 19, no. 1, p. 14.
