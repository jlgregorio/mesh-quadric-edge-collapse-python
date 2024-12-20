# Explanation

This is a tentative to explain how the quadric edge collapse algorithm works.

## Algorithm Summary

The algorithm is summarized as follows by M. Garland & P. S. Heckbert in the original paper from 1997:

1. Compute the $Q$ matrices for all the initial vertices.
2. Select all valid pairs.
3. Compute the optimal contraction target v_bar for each valid pair $(v_1, v_2)$. The error $\bar{v}^T (Q_1 + Q_2) \bar{v}$ of this target vertex becomes the cost of contracting that pair.
4. Place all the pairs in a heap keyed on cost with the minimum cost pair at the top.
5. Iteratively remove the pair $(v_1, v_2)$ of least cost from the heap, contract this pair, and update the costs of all valid pairs involving $v_1$.
