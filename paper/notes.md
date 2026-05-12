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
