# AcmeCloud Contract Summary — SLA-2026-04-A3

**Document ID:** NIMB-VEND-ACME-001
**Version:** 1.1
**Last updated:** 2026-07-31 (post APAC firm-commitment amendment)

This is a PM-level summary of the AcmeCloud relationship under contract SLA-2026-04-A3. The full legal contract is held by Asha Wickremasinghe (Head of Legal). This document captures the operational essentials.

---

## 1. Contract Identification

- **Contract reference:** SLA-2026-04-A3
- **Effective date:** 2026-04-15
- **Initial term:** 24 months (through 2028-04-14)
- **Renewal:** auto-renews 12 months unless terminated with 90 days' notice
- **Counterparty:** AcmeCloud International, B.V. (Amsterdam)
- **AcmeCloud Technical Account Manager:** Yusuf Almasi (replacing the previous TAM mid-March 2026)
- **CompactSync signatory:** Tomislav Hessford (COO)

## 2. Commercial Terms (Summary)

- **Committed-use spend:** $720,000 per year for compute + network (line L-01 in `07_budget.md`).
- **Object Storage:** $96,000 per year (line L-02).
- **AcmeDB Enterprise licences:** $240,000 per region per year. Two regions licensed (eu-central-1, us-east-1) for a total of $480,000 (line L-03).
- **Professional Services:** 1,200 hours per year at $150/hour ($180,000 — line L-04). Augmented by an additional 40 hours at no charge in compensation for the TAM rotation (use by 2026-09-30).

## 3. Capacity Reservation

- **eu-central-1:** firm commitment from 2026-05-01.
- **us-east-1:** firm commitment from 2026-05-01.
- **ap-southeast-1:** "best-effort" until 2026-09-01; firm commitment from 2026-09-01.

The APAC firm-commitment conversion was confirmed in writing on 2026-07-31 (post-Q2 steering action). This closes R-13 in the risk register.

## 4. SLAs

- **Compute availability:** 99.95% monthly uptime, measured per region.
- **AcmeDB availability:** 99.99% monthly uptime, measured per cluster.
- **Object Storage durability:** 11 nines (99.999999999%).
- **Replication SLA:** AcmeDB cross-region replication lag SLO is 2 seconds p95 over a rolling 30-day window. This is the baseline against which R-12 is measured (NOT the contractual SLA — the SLO is internal to the Data Squad, but the underlying technology must be capable of meeting it for Nimbus to ship).
- **AcmeCloud Support response:** P1 = 15 minutes 24/7; P2 = 1 hour business hours; P3 = 4 hours business hours; P4 = next business day.

## 5. Credit Terms

If SLAs are missed:

- Compute availability < 99.95% in a month → 10% credit on that month's L-01 spend.
- Compute availability < 99.0% in a month → 25% credit.
- AcmeDB availability < 99.99% in a month → 10% credit on AcmeDB licence cost for that month.
- AcmeDB availability < 99.5% in a month → 25% credit.

Credits are auto-applied to the next month's invoice.

## 6. Data Residency Commitments

- **eu-central-1:** EU-only data routing under contract clause 4.3.
- **us-east-1:** US-only data routing under contract clause 4.4.
- **ap-southeast-1:** APAC-only data routing under contract clause 4.5 (becomes effective with the firm-commitment conversion on 2026-09-01).

The per-tenant `region_hint` mechanism (D-09 deliverable, NIMB-046 / NIMB-047 / NIMB-048) is OUR responsibility on top of AcmeCloud's per-region commitments. AcmeCloud only commits to keep data in the requested region; mapping tenants to regions is on us.

## 7. Termination

- **For convenience:** 90 days' written notice.
- **For cause:** 30 days' notice with documented uncured material breach.
- **Lock-in mitigation:** the contract includes a 30-day data-export window after termination, at standard egress rates, with a maximum egress cost cap of $50,000.

## 8. Renewal

The contract auto-renews on 2028-04-14 for 12 months unless terminated. Nimbus does NOT need to address renewal — the renewal window opens in Q1 FY28 (about 16 months after programme close).

## 9. Outstanding Vendor-Side Actions

- **Confirmed 2026-07-31:** APAC firm-commitment conversion (closes R-13).
- **Pending:** R-12 mitigation collaboration — AcmeCloud is providing engineering support via the 40-hour ProServ compensation (use by 2026-09-30). PM tracks usage in `13_status_report_m4.md`.
- **Pending:** Yusuf to share AcmeDB shard re-balancing toolkit by 2026-08-31 (needed before rehearsal 2).

---

**References:** `01_project_charter.md`, `07_budget.md` (L-01–L-04), `05_risk_register.md` (R-12, R-13), `15_meeting_notes_steering_q1.md`, `16_meeting_notes_steering_q2.md`.
