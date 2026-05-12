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
are strongly associated with cross-day PER. For example, cross-day mean PER has
Spearman rho = 0.87 with absolute days from source at K=5 and K=10
(permutation p = 0.001; bootstrap 95% CIs roughly 0.73-0.93 and 0.72-0.93).
Relative covariance/CORAL distance is similarly predictive, with rho around
0.86 at K=5/10 and 0.77 at K=20.

Recoverability is weaker and more variable than degradation, but it also shows
a geometry signal. Input-layer recovery at K=5 is negatively associated with
distance from source: rho = -0.49 for absolute days (p = 0.003; bootstrap 95%
CI -0.70 to -0.19) and rho = -0.43 for relative covariance shift (p = 0.009;
bootstrap 95% CI -0.63 to -0.17). At K=20 the same direction remains but is not
significant in this small subset (rho = -0.34 for days, p = 0.089; rho = -0.29
for covariance, p = 0.154).

A median split by relative covariance distance gives an interpretable near/far
check. For K=5 input-layer calibration recovers 19.8% of the native-day gap in
near sessions versus 6.3% in far sessions, improving 18/19 near sessions and
16/20 far sessions. For K=20 the near group also has higher recovery fraction
(14.6% vs 10.3%), while K=10 is mixed. The important takeaway is therefore not
that a single threshold perfectly gates adaptation, but that drift geometry
strongly predicts decoder degradation and provides a measurable signal about
when small adapters are likely to help.

Geometry-nearest source selection gives the strongest result so far and directly
uses the geometry signal rather than only analyzing it after the fact. For each
target session, the nearest non-native source input layer is selected using
relative covariance distance computed from train-split features, then the frozen
official GRU is evaluated on validation trials with no adapter training. In the
offline setting where all other sessions are candidate sources, phoneme-weighted
PER drops to 11.38%, compared with 22.67% for the fixed middle source and 9.06%
for native-day decoding. This improves 39/41 sessions relative to the fixed
middle source, with a median selected source distance of 2 days.

A stricter past-only variant, where the selected input layer must come from an
earlier session, gives 13.58% phoneme-weighted PER on 40 evaluable sessions
(the first session has no past source). On the same 40 sessions, native-day PER
is 9.10% and the fixed middle source gives 22.76%. Past-only geometry selection
improves 34/40 sessions relative to the fixed middle source. This is a much
stronger paper result than the small learned-adapter recovery: geometry can
guide source-layer selection and recover a large fraction of the cross-day loss
without training any new parameters.

Beginning-of-day K-shot geometry selection makes the same idea more realistic.
For each target session, only the first K validation trials are used to estimate
target neural geometry; no labels are used. A past source input layer is then
selected by relative covariance distance and PER is measured only on the
remaining trials. This gives 14.00% phoneme-weighted PER at K=5, 13.19% at
K=10, and 12.74% at K=20. On the same remaining-trial subsets, fixed middle
source decoding gives 22.51%, 22.40%, and 22.30%, while native-day decoding
gives 8.95%, 8.88%, and 8.90%. Thus K-shot geometry source selection recovers
62.8%, 68.1%, and 71.3% of the fixed-source gap, improving 36/41, 36/40, and
34/37 sessions relative to fixed middle. This is now the cleanest main result:
a short unlabeled beginning-of-day neural window can choose a compatible
existing input layer and recover most of the cross-day degradation without
training new parameters.
