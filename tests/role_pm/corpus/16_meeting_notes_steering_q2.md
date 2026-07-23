# Q2 Steering Committee — Project Nimbus

**Document ID:** NIMB-STR-Q2
**Meeting Date:** 2026-07-09, 14:00–15:30 UTC
**Chair:** Tomislav Hessford (Sponsor)
**Minutes by:** PM

(Special-topic steering convened post-Q1 to review R-12 trajectory mid-quarter, per the Q1 minutes.)

## Attendees

- Tomislav Hessford (Sponsor / COO) — present
- Avantika Sundararaman (CTO) — present
- Pernille Vrieze (Head of Security) — apologies (review by written delegation)
- Bartholomew Okafor-Sing (Head of SRE) — present
- Linnaea Korhonen (Head of Customer Success) — present
- Aldous Pemberton-Riggs, Beatriz Cazadora-Olesen, Hyeon-Jin Park, Vukašin Andrássy — squad leads, present
- Yusuf Almasi (AcmeCloud TAM) — present

## Agenda + Notes

### Item 1 — R-12 trajectory (Beatriz)

Beatriz presented updated R-12 numbers:

- p95 replication lag: 5.0s → 4.5s (small improvement in M3-to-M4 transition).
- Per-shard lag instrumentation now showing two specific shards (NIMB-DB-SH-07, NIMB-DB-SH-12) account for 73% of the p95 contribution.
- Targeted parallelism tuning on those two shards expected to push p95 to ≤ 3.5s by 2026-07-31.

Sponsor: "Best estimate for hitting 3s p95 by rehearsal 2 (2026-08-21)?"
B.C-O: "70% confidence at 3s. Higher if we get the third read-replica online before then, which I expect."

### Item 2 — Approve CHG-NIMB-003 (parallelise NIMB-021 / NIMB-031)

PM presented the recovery plan for the M-LZ +1 week absorption: parallelise the AP cross-region replica work (NIMB-021) with the AP object-store replication (NIMB-031), funded by pulling one engineer from analytics (NIMB-033) for two weeks.

CTO confirmed the analytics-side slip remains inside its float.
Sponsor approved. (Formalised as DEC-2026-07-09-01.)

### Item 3 — R-19 Helmsdale acceptance escalation

Linnaea reported on the Helmsdale outreach (action from Q1 steering):

- Helmsdale's COO accepted the compressed 14-day validation window in principle on 2026-07-06.
- Helmsdale requested 1-week advance notice when D-09 delivery hits 50% confidence to mobilise their compliance team. Customer Success to track.

Sponsor formalised DEC-2026-07-09-02 (Helmsdale contingency pricing commitment).

### Item 4 — Cut-over operational readiness (Bartholomew)

Bartholomew reported on the SRE side of cut-over readiness:

- Shadow-alerting bridge (R-17 mitigation) prototype passed internal review 2026-07-04.
- Runbooks for the new estate at 60% completeness. Target 95% by 2026-09-30 (one month pre-cut-over).
- On-call rotation roster reorganised to include AcmeCloud certifications by 2026-09-15.

No decisions required.

### Item 5 — Engineering bandwidth for D-09 (CTO)

CTO formally committed engineering resources to D-09 from 2026-09-01 through 2026-11-12. Mitigation closes R-16.

PM: "What's the recourse if Q3 product work slips into this window?"
CTO: "Q3 product work has been re-baselined to land before 2026-09-01. There is no contingency to slip Q3 work onto Nimbus engineers — if Q3 slips, Q3 reschedules, not Nimbus."

### Item 6 — Communications cadence

Sponsor: "I want the next status report to surface R-12's trajectory more prominently. Yellow status means 'Sponsor needs to know what to do' and I want the recommended action to be the first paragraph."

PM accepted. M4 status (this document's reference) leads with the rehearsal-1 outcome and the R-12 trajectory ahead of all other content.

## Decisions Taken

- DEC-2026-07-09-01: approve CHG-NIMB-003 (parallelisation).
- DEC-2026-07-09-02: approve Helmsdale R-19 contingency pricing commitment.

## Actions

- PM to publish these minutes by 2026-07-11 (done).
- Customer Success to give Helmsdale 1 week's advance notice when D-09 hits 50% confidence.
- CTO to confirm Q3 product baseline by 2026-08-15.
- Data Squad to push p95 lag to ≤ 3.5s by 2026-07-31 (achieved 4.2s at month end — short of target; mitigation continues).

## Next Steering Meeting

Q3 steering: 2026-10-08, 14:00–15:30 UTC (regular cadence).

---

**References:** `13_status_report_m4.md`, `05_risk_register.md` (R-12, R-19), `09_change_log.md` (CHG-NIMB-003), `14_decision_log.md` (DEC-2026-07-09-01, DEC-2026-07-09-02).
