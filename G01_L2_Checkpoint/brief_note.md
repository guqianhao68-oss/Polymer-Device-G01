# G01 L2 checkpoint note

The 56 measurement indices each contain three paired repeats at T = 300 K. The mean Hall signal crosses zero between indices 26 and 27. Hall signs and measurement order were retained, and the two channels were normalised independently. The candidate models and numerical grids follow Practical Methods Guide pp. 6-9.

The cone RMS is 0.010803; the best hard-gap RMS is 0.015919, at Eg/kBT = 0.50. This is a -47.36% residual improvement. Applying both the 15% rule and Eg >= 1 kBT selects **cone**, with central Eg/kBT = 0.00 (Eg = 0.000000 eV).

For 200 paired repeat-resampling fits (NumPy default_rng seed 6028), the gap median is 0.00 kBT and the 16th-84th percentile interval is [0.00, 0.00] kBT. The hard-gap selection fraction is 0.0%. This fraction measures stability under the specified resampling procedure, not a posterior probability that the material is gapped. The trajectory figure reports the fit to repeat means and pointwise 16th-84th percentile intervals in the original measurement order.

Device implication: the cone result supports evaluating this material for sensing or interconnect use rather than relying on a hard band gap for switching. Hall response can track carrier changes, but a useful sensing operating point must be selected from the response curve and noise. These normalised measurements alone do not establish absolute interconnect performance or a high transistor on/off ratio. No resolved hard gap is demonstrated.

The zero-valued bootstrap gap interval reflects repeated selection of the cone under the supplied rule. It is not proof that the true gap is exactly zero, and it is not a continuous upper bound on an unresolved small gap.

The reported uncertainty is conditional on the two supplied symmetric DoS models and the finite grids. Three repeats give limited noise information. A subthermal gap may not be distinguishable from a cone. The model selection threshold and 0.25 kBT gap grid introduce discrete outcomes; the percentile band is pointwise, not a simultaneous confidence band.

Submission scope: L2 checkpoint. The supplied results_template.json keys and nesting are preserved. L1 and L3 retain their template placeholders because those phases are not populated by this L2-only pipeline. Eg_uncertainty_kT is encoded as the scalar half-width (p84-p16)/2 = 0.00; the full percentile interval is reported above. This encoding is an explicit convention because the template provides null without a type definition. The local commit identifies the code that generated these files, not student contributions or an L1-L3 blind-test pipeline.
