# Uncertainty

For class evidence `e`, the model uses `alpha=e+1`, strength `S=sum(alpha)`, and predictive mean `p=alpha/S`. Vacuity is `K/S`. Predictive entropy measures ambiguity of `p`; expected categorical entropy and their difference provide aleatoric/distributional views. The code also exposes the variance-based uncertainty measures used by the experiments.

`dirichlet_uncertainty(alpha, class_dim=1)` accepts arbitrary spatial dimensions. `pool_instance_uncertainty(alpha, labels)` sums evidence (`alpha-1`) over every positive instance before restoring the unit prior, producing per-nucleus class, confidence, and uncertainty values. Centroid-head uncertainty supports historical Beta and Normal-Inverse-Gamma outputs as well as geometric peak ambiguity.

Uncertainty values should be validated on the target domain; they are confidence diagnostics, not clinical guarantees.

