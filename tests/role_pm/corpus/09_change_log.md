# Change Log — Project Nimbus

**Document ID:** NIMB-CHG-001
**Version:** 1.4
**Last updated:** 2026-08-12 (Month 4)

Every formal change request raised against Nimbus is recorded here. IDs use the pattern `CHG-NIMB-NNN`. Status is `approved`, `pending`, `rejected`, or `withdrawn`.

---

## 1. Approved Changes

### CHG-NIMB-001 — Budget envelope raise (charter v1.1 → v1.2)

- **Raised:** 2026-04-02
- **Approved:** 2026-04-03
- **Decided by:** Tomislav Hessford (with CTO concurrence)
- **Description:** Raise the budget envelope from $2,200,000 to $2,400,000 to fund the dual-region AcmeDB Enterprise licences (line L-03 in `07_budget.md`).
- **Reference:** DEC-2026-04-02-01 in `14_decision_log.md`.
- **Status:** approved, implemented.

### CHG-NIMB-002 — Add OIDC federation to scope

- **Raised:** 2026-04-21
- **Approved:** 2026-04-23
- **Decided by:** PM (within standing authority — $0 budget impact, no schedule impact)
- **Description:** Add OIDC federation from corporate Okta to AcmeCloud IAM as part of D-03. Originally OIDC was assumed to be already in place; discovery showed it was not.
- **Impact:** added NIMB-006 to the WBS. No schedule impact (parallelisable with other Platform-squad work).
- **Status:** approved, implemented.

### CHG-NIMB-003 — Parallelise NIMB-021 with NIMB-031

- **Raised:** 2026-07-09
- **Approved:** 2026-07-09 (Q2 steering)
- **Decided by:** Tomislav Hessford + CTO
- **Description:** Recover the +1-week M-LZ slip by parallelising the cross-region replica work (NIMB-021) with the object-store replication (NIMB-031), both targeting ap-southeast-1. Required pulling one engineer from the analytics workstream (NIMB-033) for two weeks.
- **Impact:** zero net schedule impact; analytics workstream slipped by 6 working days but remains within float.
- **Status:** approved, in-flight.

### CHG-NIMB-004 — Defer customer-facing maintenance-window emails to T-21 days

- **Raised:** 2026-08-04
- **Approved:** 2026-08-06
- **Decided by:** PM + Customer Success
- **Description:** Defer enterprise-customer notification of the 2026-10-16 maintenance window from T-45 days to T-21 days, to avoid sending notifications during the August holiday slow-down when most enterprise contacts are on leave.
- **Impact:** none on the schedule.
- **Status:** approved, scheduled for 2026-09-25.

## 2. Pending Changes

### CHG-NIMB-005 — Third cut-over rehearsal contingency

- **Raised:** 2026-08-10
- **Description:** If R-12 mitigation does not bring p95 replication lag under 3 seconds by 2026-09-04, schedule a third cut-over rehearsal (in addition to the two currently planned). Estimated cost: $24,000 (drawn from the contingency reserve, line L-08).
- **Decision deadline:** 2026-09-15
- **Decision authority:** Tomislav Hessford (above PM standing authority because it touches the contingency reserve)
- **Status:** pending. Decision linked to the lag-test outcome on 2026-09-04.

## 3. Rejected Changes

### CHG-NIMB-006 — Microservices refactor

- **Raised:** 2026-05-15 (by an engineering manager not on the programme)
- **Rejected:** 2026-05-17
- **Decided by:** Avantika Sundararaman
- **Description:** Proposal to take the migration as an opportunity to refactor CompactSync from a monolith to microservices.
- **Rejection rationale:** explicitly out of scope per `02_scope_statement.md` exclusion OS-5. Adding it would extend the timeline by an estimated 8 months and the budget by an estimated $1.8M. Not approved.
- **Status:** rejected.

## 4. Withdrawn Changes

### CHG-NIMB-007 — Substitute managed Kafka for the CDC pipeline

- **Raised:** 2026-06-22
- **Withdrawn:** 2026-06-27
- **Raised by:** B.C-O (Data Squad Lead)
- **Description:** Initial proposal to use AcmeCloud managed Kafka in front of the CDC pipeline (NIMB-023) instead of the existing internal pipeline. Withdrawn after architectural review showed the existing internal pipeline meets the target throughput and the managed alternative would add $90,000 to L-01 + L-04 without benefit.
- **Status:** withdrawn (no action taken).

---

## Counts as of 2026-08-12

- Approved: 4 (CHG-NIMB-001, 002, 003, 004)
- Pending: 1 (CHG-NIMB-005)
- Rejected: 1 (CHG-NIMB-006)
- Withdrawn: 1 (CHG-NIMB-007)
- Total raised: 7

---

**References:** `01_project_charter.md`, `02_scope_statement.md`, `07_budget.md`, `14_decision_log.md`, `16_meeting_notes_steering_q2.md`.
