# Explanation

This is a tentative to explain how the quadric edge collapse algorithm works.

```
      o---o---o                  o---o---o   
     / \ / \ / \    contract    /  \ | /  \  
    o---v1--v2--o    =====>    o-----v-----o 
     \ / \ / \ /                \  / | \  /  
      o---o---o                  o---o---o  

       before                      after
```

## Algorithm Summary

The algorithm is summarized as follows by M. Garland & P. S. Heckbert in the original paper from 1997:

1. Compute the $Q$ matrices for all the initial vertices.
2. Select all valid pairs.
3. Compute the optimal contraction target $\bar{v}$ for each valid pair $(v_1, v_2)$. The error $\bar{v}^T (Q_1 + Q_2) \bar{v}$ of this target vertex becomes the cost of contracting that pair.
4. Place all the pairs in a heap keyed on cost with the minimum cost pair at the top.
5. Iteratively remove the pair $(v_1, v_2)$ of least cost from the heap, contract this pair, and update the costs of all valid pairs involving $v_1$.

## Details

### $Q$ matrices

A $Q$ matrix is a symmetric 4 x 4 matrix associated with each vertex. It is the sum of the fundamental error quadrics, which are constructed by considering the Cartesian equation of the planar faces (triangles) meeting a this vertex.

### Pair selection

A pair of vertices $(v_1, v_2)$ is a valid pair if it is an edge of the mesh or optionally if their distance is below a given threshold.
