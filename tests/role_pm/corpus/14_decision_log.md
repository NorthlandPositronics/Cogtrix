# Decision Log — Project Nimbus

**Document ID:** NIMB-DEC-001
**Version:** 1.4
**Last updated:** 2026-08-12

Every formal decision made on Nimbus is recorded here with rationale and alternatives considered. ID pattern: `DEC-YYYY-MM-DD-NN`.

---

## DEC-2026-04-02-01 — Raise budget envelope from $2,200,000 to $2,400,000

**Decision:** Raise the approved envelope to $2,400,000 to fund dual-region AcmeDB Enterprise licences.
**Owner:** Tomislav Hessford (with CTO concurrence)
**Date:** 2026-04-02
**Status:** approved, implemented (CHG-NIMB-001).

**Alternatives considered:**

1. Stay at $2,200,000 by using a single AcmeDB region with cross-region read-only replicas → rejected. Failover scenarios would force a 15+ minute promotion window; unacceptable for SC-2.
2. Use a third-party multi-region database (CockroachDB or YugabyteDB) → rejected. No precedent in our estate; learning curve incompatible with the 9-month delivery window.
3. Defer the multi-region requirement to a follow-on programme → rejected. Helmsdale (HL-2025-NOV) contract terms make multi-region mandatory by 2026-12-31.

**Rationale:** Option 1 fails SC-2. Option 2 is too risky given the timeline. Option 3 jeopardises Helmsdale renewal. The licence cost is bounded ($480,000 across two regions; the APAC region runs as a read-only replica without additional licence cost — R-18 to monitor).

---

## DEC-2026-05-10-01 — Approve M-LZ +1 week recovery plan

**Decision:** Absorb the M-LZ +1 week slip by parallelising NIMB-007 (service-account reconciliation) with NIMB-008 (observability bootstrap).
**Owner:** Tomislav Hessford
**Date:** 2026-05-10
**Status:** approved, implemented.

**Alternatives considered:**

1. Extend the programme end date by 1 week → rejected. Programme end is bounded by the Frankfurt lease termination window (charter section 11 assumption).
2. De-scope NIMB-007 → rejected. Service-account hygiene is a security prerequisite (Pernille Vrieze veto).
3. Add contractor capacity to Platform Squad → rejected. Onboarding overhead exceeds the velocity gain over a 4-week period.

**Rationale:** Parallelisation is the lowest-risk option. NIMB-007 + NIMB-008 do not contend for any shared resource.

---

## DEC-2026-06-14-02 — Release L-11 buffer to L-01 + L-04 forecast variance

**Decision:** Release the unallocated $190,000 buffer (line L-11) to cover forecast variance on L-01 (+$15,000) and L-04 (+$7,000), retaining the remaining $168,000 against unforeseen line items in M5–M9.
**Owner:** Eberhard Lindqvist-Marais (Head of Finance), PM (within standing authority since each individual line variance is under $20,000).
**Date:** 2026-06-14
**Status:** approved, implemented.

**Alternatives considered:**

1. Leave L-11 unallocated and draw down on contingency (L-08) instead → rejected. Contingency is held by the Sponsor for material risk; this is forecast smoothing, not contingency.
2. Reduce L-04 forecast by cutting AcmeCloud Professional Services → rejected. ProServ is funding R-12 mitigation, not optional.

**Rationale:** L-11 was created precisely to absorb forecast variance without needing to call down contingency. Using it for its intended purpose.

---

## DEC-2026-07-09-01 — Approve CHG-NIMB-003 (parallelise NIMB-021 / NIMB-031)

**Decision:** Parallelise the cross-region replica AP work (NIMB-021) with the object-store replication AP work (NIMB-031) by pulling one engineer from the analytics workstream (NIMB-033) for two weeks.
**Owner:** Tomislav Hessford + Avantika Sundararaman (taken at Q2 steering, 2026-07-09)
**Status:** approved, in-flight.

**Alternatives considered:**

1. Sequential delivery (the original plan) → rejected. Adds 6 working days to M-DB-R that we cannot afford.
2. Pull engineer from D-09 instead → rejected. D-09 is the Helmsdale-critical feature (R-19); cannot weaken it.
3. Pull engineer from Migration Squad (NIMB-037 / NIMB-041) → rejected. Migration Squad is the critical path for M-CUT.

**Rationale:** Analytics workstream has the most schedule float of the four candidates. NIMB-033 slips 6 working days but remains inside its float.

---

## DEC-2026-07-09-02 — Approve Helmsdale R-19 contingency commitment

**Decision:** Authorise Customer Success to offer Helmsdale Logistics a written commitment to honour 2025 contract pricing through Q1 FY27 if formal acceptance slips.
**Owner:** Tomislav Hessford (taken at Q2 steering)
**Status:** approved, on-call (not yet exercised).

**Alternatives considered:**

1. No contingency, let the renewal terms reopen → rejected. Estimated loss: $640,000 ARR.
2. Stronger commitment (e.g., service credits) → rejected. Sponsor judged the pricing-extension option sufficient given Helmsdale's relationship history.

**Rationale:** Cheapest credible commitment that retains the renewal.

---

## DEC-2026-08-06-01 — Approve CHG-NIMB-004 (defer customer notification to T-21 days)

**Decision:** Move enterprise-customer notification of the 2026-10-16 maintenance window from T-45 to T-21 days, to avoid the August holiday slow-down.
**Owner:** PM + Linnaea Korhonen
**Status:** approved, scheduled.

**Alternatives considered:**

1. Send at T-45 anyway → rejected. ~40% of enterprise customer contacts are on August leave; communication risks getting lost.
2. Phased notification (T-45 to "champion" contacts, T-21 to general) → rejected. Adds operational complexity for low marginal benefit.

**Rationale:** T-21 gives enterprise customers 3 working weeks to plan around the cut-over, which is the SLA-required notice period.

---

## Open Decisions

### D-BUD-2026-08-1 — Third cut-over rehearsal contingency (CHG-NIMB-005)

**Status:** pending. Decision deadline 2026-09-15. Trigger: rehearsal 2 outcome on 2026-08-21. Decision authority: Tomislav Hessford.

---

**References:** `01_project_charter.md`, `07_budget.md`, `09_change_log.md`, `16_meeting_notes_steering_q2.md`.
