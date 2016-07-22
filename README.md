# GSDMM (short text topic modeling) 

## A python implementation of http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf)

## Usage: `python test_lda.py`

Note: In `test_lda.py` package `lda` is required (`pip install lda`). However, it is unnecessary to install it to use `GSDMM.py`.

Note: The `X` matrix can be either a count matrix, where `X[document, word]` denotes the word count, or a binary matrix, where `X[document, word]=1/0`, indicating whether a word occurs or not.
