# Q1 Steering Committee — Project Nimbus

**Document ID:** NIMB-STR-Q1
**Meeting Date:** 2026-06-27, 14:00–15:30 UTC
**Chair:** Tomislav Hessford (Sponsor)
**Minutes by:** PM

## Attendees

- Tomislav Hessford (Sponsor / COO) — present
- Avantika Sundararaman (CTO) — present
- Pernille Vrieze (Head of Security) — present
- Bartholomew Okafor-Sing (Head of SRE) — present
- Linnaea Korhonen (Head of Customer Success) — present
- Eberhard Lindqvist-Marais (Head of Finance) — observer
- Asha Wickremasinghe (Head of Legal) — observer (joined for items 4 and 5)
- Aldous Pemberton-Riggs, Beatriz Cazadora-Olesen, Hyeon-Jin Park, Vukašin Andrássy — squad leads, present
- Yusuf Almasi (AcmeCloud TAM) — present for items 3 and 4

## Agenda + Notes

### Item 1 — Programme status (PM)

PM walked through the M3 status report. Headlines:

- M-LZ delivered on recovered plan (2026-05-29).
- M-NET on track (2026-06-19 target).
- Budget cumulative variance -5.2% (favourable).
- R-12 progressing (8s → 5s p95 replication lag).

No decisions required.

### Item 2 — Risk-register deep-dive (Beatriz)

Beatriz presented R-12 in detail. Mitigation actions in flight:

- Tune AcmeDB replication parallelism (4 → 8). Implemented 2026-06-12. Result: lag dropped from 8s to 5s p95.
- Add two read-replica nodes in us-east-1. Provisioning in progress; live 2026-07-15.
- Per-shard lag metrics. Instrumentation merged 2026-06-20.

Sponsor question: "What's the failure mode if we hit rehearsal 1 with lag still above 3s?"
B.C-O: "We'd request a third rehearsal under CHG-NIMB-005. The contingency is already documented; cost $24,000."
Sponsor: "Acknowledged. We will not pre-approve the third rehearsal — make the call based on rehearsal 1's outcome."

### Item 3 — AcmeCloud relationship review (Yusuf)

Yusuf reported:

- The Frankfurt TAM rotation cost the programme an estimated 5 days. Compensation: AcmeCloud has offered an additional 40 hours of ProServ at no charge, valid through 2026-09-30.
- The APAC capacity reservation (R-13) is on track to convert from "best-effort" to firm commitment on 2026-09-01.

Sponsor accepted the ProServ compensation. PM to use it against the R-12 mitigation work.

### Item 4 — Helmsdale acceptance window (Linnaea, Asha)

Linnaea introduced R-19. The risk: Helmsdale's 21-day contractual validation window falls inside the holiday slow-down (2026-11-12 → 2026-12-03). Their compliance team operates at ~50% capacity during this window.

Linnaea's proposal: ask Helmsdale's COO for a written commitment to compress their validation to 14 days, conditional on us delivering D-09 one week ahead of plan.

Asha confirmed: from a contracts standpoint, this is a low-risk request — it does not modify the master contract, only the operational schedule.

Sponsor approved:

- Linnaea to make the request to Helmsdale's COO by 2026-07-10.
- If Helmsdale declines the compressed window, Customer Success may offer the 2025 contract-pricing extension through Q1 FY27 as a contingency (DEC-2026-07-09-02 formalised this two weeks later at the Q2 steering — the Sponsor took the approval as a verbal pre-commit here and the formal log entry post-dates this meeting).

### Item 5 — Programme communication review

PM walked through the communication plan (`18_communication_plan.md`). No changes requested.

Quentin Ostrowski (not in attendance) had been briefed in advance and his enterprise-customer notification template was reviewed and approved.

## Decisions Taken

- Sponsor: do NOT pre-approve third cut-over rehearsal. Decision to be taken after rehearsal 1.
- Sponsor: accept AcmeCloud's 40-hour ProServ compensation against R-12 mitigation work.
- Sponsor (verbal pre-commit): authorise Customer Success to offer Helmsdale 2025 contract pricing through Q1 FY27 if D-09 slips. Formalised as DEC-2026-07-09-02 at Q2 steering.

## Actions

- Linnaea to contact Helmsdale's COO by 2026-07-10 with the compressed-validation-window request.
- PM to use the 40-hour ProServ compensation for R-12 mitigation (logged against L-04).
- Beatriz to publish per-shard lag dashboard by 2026-07-04 (done).
- PM to publish these minutes by 2026-07-04 (done).

## Next Steering Meeting

Q2 steering: 2026-07-09, 14:00–15:30 UTC (one round of regular steerings then a special-topic steering planned because R-12 trajectory needs a mid-quarter review).

---

**References:** `12_status_report_m3.md`, `05_risk_register.md`, `09_change_log.md`, `14_decision_log.md`.
