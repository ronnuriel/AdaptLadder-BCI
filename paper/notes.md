# Paper Notes

Native-day evaluation confirms that the official GRU checkpoint is reproduced
correctly, reaching 9.06% phoneme-weighted PER on the T15 validation split. A
cross-day stress test, forcing all validation sessions through the early
`t15.2023.08.13` input layer, increases PER to 28.81% and harms 40/41 sessions.
This establishes a decoder-facing cross-session drift effect and motivates
source-sweep and lightweight adaptation experiments.

A small source sweep shows that this effect is not unique to the early source
layer. Using `t15.2023.11.26` as a middle source gives 22.67% phoneme-weighted
PER, while the late `t15.2025.04.13` source gives 34.43%. All three non-native
sources harm 40/41 validation sessions, supporting the claim that mismatched
day-specific input layers cause broad decoder-facing degradation.

The first adaptation ladder experiment used the middle `t15.2023.11.26` source
layer and full-session unsupervised validation statistics. Cross-day no
correction reproduced the 22.67% phoneme-weighted PER baseline exactly. Target
z-scoring gave 22.75% PER and moment matching to the source distribution gave
22.73% PER, with no positive recovery of the native-day gap. This suggests that
simple channel-wise mean/variance alignment is not enough to emulate the
official day-specific input layers; those layers likely capture stronger
session-specific transformations than first and second moments.

The next ladder step is supervised small-calibration diagonal affine adaptation.
For each target session, the first K labeled validation trials are used only to
learn per-channel scale and bias through CTC loss while freezing the official
GRU and source input layer. PER is then measured on the remaining trials. This
keeps the intervention lightweight while testing whether minimal learned
feature reweighting can recover what unsupervised moment correction could not.

A CPU-feasible all-session pilot with 5 adapter-training epochs gives a positive
but modest signal. For the middle `t15.2023.11.26` source, diagonal affine
improves weighted PER from 22.51% to 21.99% at K=5, from 22.48% to 21.81% at
K=10, and from 21.51% to 20.21% at K=20. Recovery of the native-day gap rises
from 3.8% to 4.9% to 10.1%, while moment matching remains worse than no
correction. Because this run used only 5 epochs, it should be treated as a
preliminary full-session result rather than the final calibration setting.

A full input-layer calibration variant was also tested with the same first-K
protocol, freezing the GRU and output head while learning a 512x512 input matrix
and bias initialized from the middle source input layer. With 5 training epochs,
input-layer calibration improves weighted PER from 22.51% to 21.70% at K=5,
from 22.48% to 21.60% at K=10, and from 21.51% to 20.27% at K=20. This is
slightly better than diagonal affine at K=5/10 and roughly tied at K=20, but it
still recovers only about 6.0%, 6.5%, and 9.7% of the native-day gap. The main
takeaway is therefore not that full input-layer calibration solves drift, but
that learned small-calibration adapters consistently outperform moment matching
while remaining far from native day-specific adaptation.

Recovery-geometry analysis was added to connect the adaptation results with the
shape of cross-session drift. Using sampled train frames relative to the middle
`t15.2023.11.26` source, temporal distance and covariance/CORAL-style distance
are strongly associated with cross-day PER (Spearman correlations around
0.75-0.87 across K subsets). Input-layer recovery is weaker for sessions that
are farther away in time or covariance/subspace geometry; for example, K=5
input-layer recovery has Spearman correlations of -0.49 with absolute days from
source and -0.43 with relative covariance shift. This supports the emerging
interpretation that small adapters help most when the target session remains
geometrically close enough to the source, while larger drift may require a
stronger adaptation level.
