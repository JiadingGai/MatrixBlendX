Project structure based on torch cpp_extension tutorial.

1. The jit compile approach, simply run:
python blend_jit_compile.py 

2. The setuptools approach:
  Step 1. install blend_cpp module: 
    python setup.py install
  Step 2. run test: 
    python blend_setuptools.py 

**Flash attention math**

Consider $Q$ with a single row block of $Q_1 \in R^{B_r \times d}$: $$Q = \big[Q_1,\big]$$
and $K^T = \big[K_1^T,\ K_2^T\big] \in R^{d \times 2B_c}$, where $K_i \in R^{B_c \times d}$.

Then, $$S = \big[S_1,\ S_2\big] = \big[Q_1K_1^T,\ Q_1K_2^T\big] \in B_r \times 2B_c$$.

Multiply with 

$$
V=
\begin{bmatrix}
  V_1 \\
  V_2
\end{bmatrix},\ \text{where}\ V_i \in R^{B_c \times d}
$$

* Standard attention compute:

  $m = \max(\text{rowmax}(S_1), \text{rowmax}(S_2)) \in R^B_r$

  $l = \text{rowsum}(e^{S_1-m}) + \text{rowsum}(e^{S_2-m}) \in R^B_r$

  $P=[P_1,\ P_2] = \text{diag}(l)^{-1} \bigg[e^{S_1-m},\ e^{S_2-m}\bigg] \in R^{B_r \times 2B_c}$

  $O = [P_1,\ P_2] \times [V_1,\ V_2] = \text{diag}(l)^{-1} \bigg[e^{S_1-m} V_1 + e^{S_2-m} V_2 \bigg] \in R^{B_r \times d}$

**2D rotary embedding**: 
* Complex representation:
  $$(x_1 + jx_2) \cdot (\text{cos}m\theta_1 + j  \text{sin}m\theta_1)$$
  
* Cartesian representation:
  
$$
\begin{bmatrix}
  \text{cos}m\theta_1 & -\text{sin}m\theta_1\\
  \text{sin}m\theta_1 & \text{cos}m\theta_1
\end{bmatrix}
\begin{bmatrix}
  x_1 \\
  x_2
\end{bmatrix}
\=
\begin{bmatrix}
  x_1 \\
  x_2
\end{bmatrix} \odot 
\begin{bmatrix} 
   \text{cos}m\theta_1 \\
   \text{cos}m\theta_1
\end{bmatrix}
+
\begin{bmatrix}
  -x_2 \\
   x_1 
\end{bmatrix} \odot 
\begin{bmatrix} 
   \text{sin}m\theta_1 \\
   \text{sin}m\theta_1
\end{bmatrix}
$$
