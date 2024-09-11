Project structure based on torch cpp_extension tutorial.

1. The jit compile approach, simply run:
python blend_jit_compile.py 

2. The setuptools approach:
  Step 1. install blend_cpp module: 
    python setup.py install
  Step 2. run test: 
    python blend_setuptools.py 

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
