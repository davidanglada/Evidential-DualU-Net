# Uncertainty

For class evidence `e`, the model uses `alpha=e+1`, strength `S=sum(alpha)`, and predictive mean `p=alpha/S`. The paper defines aleatoric uncertainty as `sum_k alpha_k(S-alpha_k) / [S(S+1)]`, epistemic uncertainty as `sum_k alpha_k(S-alpha_k) / [S²(S+1)]`, and vacuity as `K/S`. Their analytic extrema normalize all three to `[0,1]`. The authors explicitly interpret these as empirically useful uncertainty proxies, not strictly Bayesian quantities.

`dirichlet_uncertainty(alpha, class_dim=1)` accepts arbitrary spatial dimensions. `pool_instance_uncertainty(alpha, labels)` follows the paper: it averages Dirichlet parameters inside each positive instance, removes background class 0, and renormalizes over foreground classes. Mean pooling is size-invariant and is the reported method; `median` and `sum` remain available to reproduce Appendix C.

For a predicted Gaussian centroid map `g` and instance `Omega`, peak uncertainty is `1-max(g)` and mass-ratio uncertainty is `|sum(g)-2*pi*sigma²|/(2*pi*sigma²)`. The combined score is `lambda_peak*u_peak + lambda_mass*u_mass`; paper experiments use `sigma=5`, `lambda_peak=0.3`, and `lambda_mass=0.6`. Call `geometric_centroid_uncertainty` after undoing any training-time map scaling.

Uncertainty values should be validated on the target domain; they are confidence diagnostics, not clinical guarantees.
