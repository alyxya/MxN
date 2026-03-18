# Newton Step in Low-Rank Form

After algebra, the updated matrix still has the form

$$
W' = I + A'B'
$$

where

$$
A' = A - \frac{1}{2} A(BB^T) - \frac{1}{2} A(A^T A)(BB^T)
$$

and

$$
B' = B
$$

or, symmetrically, you can push updates into $B$ instead.

The important thing: all expensive operations are $k \times k$.

Cost:

$$
O(nk^2)
$$

instead of

$$
O(n^3)
$$
