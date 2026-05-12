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
