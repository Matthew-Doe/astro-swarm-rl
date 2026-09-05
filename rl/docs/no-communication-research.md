# No-communication interceptor research ideas

## Research constraint

Each interceptor must act from its own local history. There is no radio, shared blackboard, teammate message, global tactical map, or runtime parameter update. A shared neural policy is allowed because it is fixed before deployment; shared weights are not communication.

The current observation design is compatible with this constraint: each interceptor receives its own state and entities inside its local sensor cone. Seeing another vehicle is environmental sensing, not a message. The attacker may retain global defender visibility in the advantaged-attacker curriculum, since that makes the interceptor problem harder.

The centralized value function used during training is a CTDE (centralized-training/decentralized-execution) aid. For a stricter study, train a second condition with a local-only value function and report the difference.

## Strong attacker starting condition

Use a fixed, reproducible pressure-test profile before introducing the full randomized distribution:

| Variable | Pressure-test setting |
| --- | ---: |
| Interceptors | 2, then 3 |
| Attacker speed | 185–200 px/s |
| Interceptor speed | 130–155 px/s |
| Attacker turn rate | 3.5–4.0 rad/s |
| Interceptor turn rate | 2.8–3.4 rad/s |
| Interceptor FOV | 25–45° |
| Interceptor view range | 150–220 px |
| Attacker spawn radius | 700–800 px |
| Attacker information | Exact defender positions and visibility |

The initial target should be a 15–30% interceptor win rate. This gives the decentralized team useful failures without making positive learning signal disappear.

## Reward candidates

Keep rewards zero-sum and log every component independently.

### Team outcome

- Capture: `+1.0` to interceptors
- Goal: `-1.0` to interceptors
- Timeout: small interceptor win or configurable draw, depending on the research question

### Local interception skill

- First local detection: `+0.05`
- Maintaining a valid local track: `+0.001` per step, capped per episode
- Closing distance along a predicted intercept course: small potential-based reward
- Capture credit: larger share to the interceptor that made contact

### Team behavior without messages

Reward observable behaviors, not hidden coordination variables:

- Different interceptors covering different angular sectors
- One interceptor cutting off the predicted route while another searches
- Maintaining useful spacing
- Avoiding redundant pursuit by all vehicles
- Penalizing planet camping and stationary overlap

These terms should be small compared with capture/goal rewards. Otherwise the team can maximize formation or coverage while allowing the attacker through.

### Attacker pressure

For the advantaged attacker, make detection costly over time rather than only at the terminal frame. A possible attacker reward is:

```text
goal while never detected: +1.5
goal after detection:       +0.5
capture:                    -1.0
per-step exposure cost:     -0.002 × visible-defender fraction
goal-progress shaping:      existing bounded potential
```

The exposure term should be capped or normalized so it cannot outweigh reaching the goal.

## Ablation matrix

Run the same seeds across these conditions:

1. Local observations, no communication, centralized critic
2. Local observations, no communication, local critic
3. Local observations plus teammate sensing inside the cone
4. Same as (3), with teammate sensing removed entirely
5. Normal attacker versus advantaged attacker
6. Two, three, and five interceptors

This distinguishes genuine decentralized sensing from benefits caused by centralized training or population size.

## Metrics to report

- Interceptor win rate and attacker goal rate
- Capture time and time-to-first-detection
- Percentage of goals reached undetected
- Number of active interceptors near capture
- Pairwise distance and angular coverage
- Duplicate-pursuit rate
- Performance after removing one interceptor
- Robustness to sensor noise, missed detections, and spawn changes
- Performance against unseen attacker policies

The most important success criterion is not merely win rate: a good no-communication team should continue to intercept when formations, seeds, and attacker policies change.

## Suggested implementation order

1. Add the pressure-test simulator profile and keep the normal profile unchanged.
2. Add reward-component telemetry and episode-level metrics.
3. Add local-history stacking or a recurrent actor if instantaneous observations are insufficient.
4. Add coverage, predicted-intercept, and duplicate-pursuit terms one at a time.
5. Compare centralized-training and local-training critics.
6. Evaluate only frozen actors in Godot, with no communication hooks compiled into the runtime path.

The existing anti-camping rule and reduced balanced-profile FOV are a useful baseline. The next experiment should change only the attacker handicap and one reward term at a time so improvements can be attributed.
