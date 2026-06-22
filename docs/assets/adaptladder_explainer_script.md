# AdaptLadder-BCI narrated explainer script

## Slide 1: Open with the question

I want to start with a practical question. Do we really need to recalibrate or retrain a brain-to-text decoder every day? In a B C I system this is not just theoretical. Daily calibration costs user time and requires labeled data. Our main answer is: not simply yes and not simply no. The main decoder can stay frozen, but the input state for the current day must be chosen carefully.

## Slide 2: Daily calibration burden

Daily recalibration can become a user burden. Every calibration block is time the user spends maintaining the system instead of communicating. Supervised updates also require known prompts or labels at the start of the day. So the practical question is not only whether we can retrain. The question is whether we can keep the decoder fixed and manage the input state instead.

## Slide 3: Reframe the problem

This reframes the daily problem. Instead of asking: how do we train the model again today, we first ask: which historical input state is compatible with today's brain signal? The project does not propose a new speech decoder. It studies how to maintain an existing decoder by selecting the right day-specific input layer.

## Slide 4: Frozen GRU, changing input adapter

The model we study is the public T fifteen brain-to-text G R U. The G R U backbone and phoneme output head stay frozen. What changes across days is the neural activity entering the model. The input state is the day-specific input layer, or adapter. It maps today's neural features into the representation expected by the frozen decoder.

## Slide 5: Wrong input state breaks decoding

The first result shows why input-state choice matters. With the correct native-day input layer, decoding reaches about nine point zero six percent P E R. P E R means phoneme error rate; lower is better. But when sessions are forced through a wrong non-native input layer, P E R jumps to roughly twenty two to thirty four percent. The frozen decoder is not necessarily broken. The interface into it is wrong.

## Slide 6: Moment correction and small calibration

We then asked whether the drift is just a simple mean or scale shift in individual channels. Moment matching corrects channel means and variances, but it stays near twenty one point eight two percent P E R. Input-layer calibration with twenty labeled trials helps, reaching eighteen point three eight percent. But it remains far from native-day performance, so labels help, but they should not always be the first move.

## Slide 7: Select a compatible past state

The strongest low-cost move is to select a compatible input state from history. In the K equals twenty beginning-of-day setting, reusing the previous input state reaches twelve point two one percent P E R. Geometry-based retrieval is close, at twelve point seven four percent. This leads to the core ladder: reuse previous, retrieve by geometry, and calibrate only if needed.

## Slide 8: Neural geometry explanation

Neural geometry means the shape of the neural activity. The data has trials, time bins, and neural features. We flatten trials and time so that each time bin becomes one point in a high-dimensional channel space. A session becomes a cloud of points. Covariance summarizes how channels move together. If today's cloud resembles a past day, that past input state may still be compatible.

## Slide 9: Adaptation ladder Figure 1

This figure summarizes the method. At the start of a new day, a short K-shot window estimates target-session geometry. For geometry retrieval, no sentence labels are needed. The system compares the new session to stored input states, then chooses the cheapest reliable action: reuse the previous state, retrieve an older compatible candidate, or calibrate the input layer only. Throughout this process, the G R U backbone and phoneme head remain frozen.

## Slide 10: Drift is structured

The geometry results show that drift is structured, not random. Nearby days are often nearby in geometry, creating a diagonal band in the drift map. As the number of days between source and target increases, covariance shift also tends to increase, with Spearman rho around zero point eight five in T fifteen. This explains why yesterday is such a strong baseline, while still leaving room for older states in non-monotonic cases.

## Slide 11: Geometry is not an autopilot

But geometry is not a complete policy by itself. With K equals twenty, geometry selected the previous source in twenty six sessions and an older non-previous source in eleven sessions. Among those eleven older choices, the older state actually won only three times. Previous still won seven times, and one case was tied. Geometry should propose candidates and warn about risk, not blindly override recency.

## Slide 12: Operational answer

The operational policy is simple. If the previous state is recent, reuse it. For zero to three day gaps, previous-state reuse is near native, around nine point nine seven percent P E R. If the previous state is stale, use geometry to inspect older candidates. If no stored state appears reliable, then perform input-layer calibration as an escalation.

## Slide 13: Close

So the answer is not to recalibrate by ritual. Cross-session drift can be framed as input-state management. Reuse when memory is reliable. Retrieve when geometry warns. Calibrate only when evidence demands it. The decoder can stay frozen. What needs to adapt is the input state. Thank you.

