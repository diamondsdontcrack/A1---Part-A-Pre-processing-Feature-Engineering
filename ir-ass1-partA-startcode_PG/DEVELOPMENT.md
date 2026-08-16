# Engineering Development Evidence

Paste two separate excerpts: one failed output from the provided self-checker, and one error message or incorrect program output encountered during implementation or debugging. For each, include the failed output, briefly state what you changed, and show the rerun result.

This file concerns engineering development evidence, not method design or experimental evaluation.


# Engineering Development Evidence

## Excerpt 1 — Failed self-check: `tfidf_variants(..., tf_mode='log'/'bm25')`

While implementing Task 3, the provided self-checker reported:

```text
FAIL - Task3: tfidf_variants(..., tf_mode='log'/'bm25') runs + returns correct shapes
       KeyError: 'cat'
```

The problem was in my code where I did
vocab = {}
for index, term in enumerate(terms):
       vocab = {term: index}

and later

df = {term: 0}

I had intended to add each term to vocab but the last line creates a new dictionary and replaces the previous one on each iteration leaving a dictionary with only 1 term. Because df was repeatedly replaced, by the time df[term] +=1 was executed, most keys did not exist, including 'cat'.
The change was changing vocab = {term: index} to vocab[term] = index, and df = {term:0} to df[term] = 0

This will add or update the key-value pair inside the existing dictionary and builds it progressively.

After the change, I reran the self-checker:

```text
PASS - Task3: tfidf_variants(..., tf_mode='log'/'bm25') runs + returns correct shapes
       log/bm25 produced finite matrices with consistent shapes
```

## Excerpt 2 — Incorrect Task 1 retrieval output

My initial preprocessing pipeline decoded HTML entities, extracted readable text using BeautifulSoup, lowercased it, and tokenized it with `nltk.word_tokenize`. The function ran successfully, but the public retrieval output showed that Profile B was not retrieving any relevant documents:

```text
PASS - Task1: preprocess() two-profile dev retrieval report
       Profile A: Hit@1=0.091, Hit@5=0.909;
       Profile B: Hit@1=0.000, Hit@5=0.000;
       combined Average Hit@K=0.250
```

I inspected the raw Profile B strings using `repr()` and found invisible Unicode formatting characters embedded inside words. A query contained `U+200B` ZERO WIDTH SPACE, while inspection of a relevant document also revealed `U+00AD` SOFT HYPHEN. Both belong to Unicode category `Cf`.

I changed preprocessing to remove characters belonging to the `Cf` formatting category before tokenization, rather than handling only one specific Unicode code point.

After rerunning the checker:

```text
PASS - Task1: preprocess() two-profile dev retrieval report
       Profile A: Hit@1=0.091, Hit@5=0.909;
       Profile B: Hit@1=0.091, Hit@5=0.909;
       combined Average Hit@K=0.500
```

This removed the retrieval difference between the two public noise profiles.

It should be noted that Hit@1 scores cannot be improved as they are largely due to tied token-overlap scores, including cases where the relevant document already matched every query token. And since ranking and document-ID tie-breaking are performed outside of preprocess and we are not allowed to alter the checker itself (where the ranking system exists), Hit@1 likely cannot be improved.
