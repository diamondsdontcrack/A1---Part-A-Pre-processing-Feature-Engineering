# Engineering Development Evidence

Paste two separate excerpts: one failed output from the provided self-checker, and one error message or incorrect program output encountered during implementation or debugging. For each, include the failed output, briefly state what you changed, and show the rerun result.

This file concerns engineering development evidence, not method design or experimental evaluation.


# Engineering Development Evidence

## Excerpt 1 — Failed self-check: `make_positions`

While implementing Task 2, the provided self-checker reported that `make_positions()` was not returning the required dictionary:

```text
FAIL - Task2: make_positions(...) public example
       AssertionError: Expected dict, got <class 'NoneType'>
```

The function still contained its starter `pass`, so Python implicitly returned `None`. I implemented the position mapping by iterating through the token list with `enumerate()` and appending each 0-based index to the corresponding token.

After the change, I reran the self-checker:

```text
PASS - Task2: make_positions(...) public example
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
