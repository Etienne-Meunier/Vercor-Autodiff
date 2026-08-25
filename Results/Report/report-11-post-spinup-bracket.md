# Report 11: Where Is the Trustworthy Window Post-Spinup?

Report 10 found n=20 from a 50-day-spun-up atmosphere is already unreliable (87.5% FD rel error, vs. 7.3% cold-start). This brackets smaller n from the same spun-up state to find where the trustworthy threshold actually sits.

Script: `Results/Report/scripts/report-11/post_spinup_bracket.py`.

## Setup

Same 50-day forward-only spin-up as report-10, same FD-vs-AD spot-check methodology as report-3 (single `value_and_grad` w.r.t. c_k, central FD at eps=1e-2), swept over n = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20].

## Results

| n | AD grad | FD grad | rel err |
|---|---|---|---|
| 1 | 1.0871e+02 | 1.0958e+02 | 0.8% |
| 2 | 2.7093e+02 | 2.7445e+02 | 1.3% |
| 3 | 3.9469e+02 | -4.2036e+02 | 193.9% |
| 4 | 4.0970e+02 | -7.5099e+02 | 154.6% |
| 5 | 2.7772e+02 | -9.5789e+02 | 129.0% |
| 6 | 8.4037e+01 | 9.3542e+02 | 91.0% |
| 8 | -2.6539e+02 | 4.5783e+02 | 158.0% |
| 10 | -3.2886e+01 | -2.9510e+03 | 98.9% |
| 12 | -1.7983e+02 | -7.3462e+03 | 97.6% |
| 15 | -1.2482e+03 | -5.2358e+03 | 76.2% |
| 20 | -1.4937e+03 | -1.1930e+04 | 87.5% |

Trustworthy only at n=1-2 (rel err <2%). Breaks completely by n=3 (193.9%, sign flip) and stays broken (76-194% rel err, several sign flips) through n=20.

## Comparison to report-3 (cold start)

| | report-3 (cold start) | report-11 (post-50-day spinup) |
|---|---|---|
| trustworthy through | n=20 (7.3% rel err) | n=2 (1.3% rel err) |
| breaks down by | n=25 (93% rel err) | n=3 (193.9% rel err) |
| AD gradient behavior | smooth, tight band (-4600 to -5700) across n=12-25 | erratic -- changes sign twice across n=1-20 (positive n=1-6, negative n=8-20), no smooth trend |

## Interpretation

The trustworthy window post-spinup is roughly 10x shorter than cold-start (n<=2 vs n<=20). This also changes what report-3 called a distinguishing signature: there, the AD gradient itself stayed smooth and well-behaved even past the point where FD stopped validating it, and only the FD comparison broke down. Here, the AD gradient itself is no longer smooth -- it changes sign twice across the swept range, meaning the true tangent-linear sensitivity is itself dominated by local chaotic stretching from n=3 onward, not just the FD-vs-AD comparison. Cold start's smooth-AD/broken-FD split was itself a property of starting in a quiescent regime (report-9: near-zero forcing at day 0, ramping up over ~50-90 days) where the early tangent-linear dynamics are simple; once the trajectory is already on the chaotic attractor, both AD and FD degrade together, much sooner.

## Caveats

Single run, single spin-up length (50 days), single seed. n=7, 9, 11, 13-14, 16-19 not sampled -- the exact breakpoint between n=2 and n=3 is not bracketed further. Only c_k tested (not c_eps). No repeat at a different post-spinup start date to check whether n<=2 generalizes or is specific to this particular spun-up state.
