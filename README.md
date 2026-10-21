# Quaternion Generalized Inverses in MATLAB

MATLAB code for the accompanying paper on quaternion outer inverses with prescribed subspace properties. Implementations use direct QTFM arithmetic and complex representations, with numerical comparisons, color image restoration, and Lorenz-system examples.

## Requirements

- MATLAB; reported experiments used R2024b.
- Quaternion Toolbox for MATLAB (QTFM). 
- Image Processing Toolbox for image restoration.

Keep all supplied `.m` files together, set their folder as MATLAB's Current Folder, and ensure QTFM is on the path.

## Methods

| Method | Outer inverse | Pseudoinverse |
| --- | --- | --- |
| Algorithm 1: QTFM SVD | `QTFM_X_compute` | `QTFM_pinv_compute` |
| Algorithm 2: complex SVD | `Complex_X_compute` | `Complex_pinv_compute` |
| Algorithm 3: QTFM full-rank factorization | `QTFM_X2L_compute` | `QTFM_pinv_compute_FR` |
| Algorithm 4: complex full-rank factorization | `Complex_X2L_compute` | `Complex_pinv_compute_FR` |

QTFM `pinv` provides the toolbox baseline. Supporting files implement SVD inverse construction, full-rank factorization, structure-preserving SVD, and numerical rank diagnostics. The SVD inverse constructions use `K=L=M=0`.

## Run numerical examples

Run each script separately in MATLAB:

```matlab
Example_Right_Outer_Comparision
Example_Left_Outer_Comparison
Example_Pinv_Comparision
```

The right and left examples use `k=5:5:100`, with seeds 42 and 321. The pseudoinverse example uses `k=20:20:100`, with seed 111. All use `m=3*k`, `n=2*k`, and 50 repeated evaluations on the same inputs per size.

Results include elapsed times, residuals, rank diagnostics, numerical data, and PNG/EPS plots. Output folders are:

```text
Example_Right_Outer_Comparision_output/
Example_Left_Outer_Comparison_output/
Example_Pseudoinverse_Comparison_output/
```



## Tolerance policy

For an `a`-by-`b` quaternion matrix `M`, the scale-aware cutoffs are:

```text
SVD:     max(a,b) * eps * sigma_max(M)
Pivot:   max(a,b) * eps * max_ij |M_ij|
Reduced: r * eps * sigma_max(R)
```

Here `r` is the quaternion dimension of the reduced matrix. Corresponding SVD methods share one cutoff for `W`; corresponding FR methods share one pivot cutoff and use individual default reduced cutoffs. QTFM `pinv` receives a cutoff for `A`. In the SVD pseudoinverse construction, `W=A'*A*A'`, so its cutoff applies to `W`. 

## Image restoration

Place these RGB images in `Input_images/` and run the matching scripts:

| Input image | Dimensions | Script |
| --- | --- | --- |
| `parrot512.png` | 512 × 512 | `Example_Parrot_restoration` |
| `ship7.png` | 512 × 768 | `Example_Ship_restoration` |
| `scenery.png` | 2400 × 2400 | `Example_Scenary_restoration` |
| `elephant.png` | 2482 × 3374 | `Example_elephant_restoration` |

 Images and results are saved in the corresponding `<Image>_Image_Restoration` folder. 

## Lorenz examples and attribution

`lorentz_final`  fit noisy Lorenz-system data using seed 321. 


