# RACI Matrix — Project Nimbus

**Document ID:** NIMB-RACI-001
**Version:** 1.1
**Approved:** 2026-04-15 (NIMB-050)

Each major workstream lists its **R**esponsible, **A**ccountable, **C**onsulted, and **I**nformed parties. One Accountable per row (Accountability is non-shareable). Responsibility may be shared across squads.

Names cross-reference `06_stakeholder_register.md`. Squad leads are abbreviated: A.P-R (Aldous Pemberton-Riggs, Platform), B.C-O (Beatriz Cazadora-Olesen, Data), H-J.P (Hyeon-Jin Park, Migration), V.A (Vukašin Andrássy, Networking).

---

## 1. Workstream RACI

| Workstream | R (Responsible) | A (Accountable) | C (Consulted) | I (Informed) |
|---|---|---|---|---|
| Landing zone (D-01) | A.P-R, Platform Squad | Tomislav Hessford | Pernille Vrieze, Yusuf Almasi | CTO, SRE Head, Customer Success |
| Inter-region networking (D-02) | V.A, Networking Squad | Tomislav Hessford | A.P-R, B.C-O, Yusuf Almasi | SRE Head, Customer Success |
| Identity (D-03) | A.P-R, Platform Squad | Pernille Vrieze | CTO, SRE Head | Customer Success, Legal |
| Database migration (D-04) | B.C-O, Data Squad | Avantika Sundararaman | Pernille Vrieze, SRE Head, Yusuf Almasi | Sponsor, Customer Success |
| Object store (D-05) | B.C-O, Data Squad | Avantika Sundararaman | Pernille Vrieze, A.P-R | Sponsor |
| Application migration (D-06) | H-J.P, Migration Squad | Avantika Sundararaman | A.P-R, B.C-O, V.A | Sponsor, SRE Head, Customer Success |
| Search & analytics (D-07) | H-J.P, Migration Squad | Avantika Sundararaman | B.C-O | SRE Head, Customer Success |
| Observability (D-08) | A.P-R, Platform Squad | Bartholomew Okafor-Sing | All squad leads | Sponsor |
| Region pinning (D-09) | H-J.P, Migration Squad | Avantika Sundararaman | Pernille Vrieze, Linnaea Korhonen, Sebastián Roiglund-Faye | Sponsor, Customer Success, Legal |
| Cut-over & decommission (D-10) | All squads (PM orchestrates) | Tomislav Hessford | All squad leads, Yusuf Almasi, SRE Head | Customer Success, Communications |

## 2. Programme-Level Activities

| Activity | R | A | C | I |
|---|---|---|---|---|
| Risk register maintenance | PM | Tomislav Hessford | All squad leads, risk owners | Steering committee |
| Budget tracking | Eberhard Lindqvist-Marais | Tomislav Hessford | PM | CTO, all squad leads |
| Status reporting (monthly) | PM | Tomislav Hessford | All squad leads | All stakeholders |
| Steering committee (quarterly) | PM | Tomislav Hessford | CTO, all squad leads, Head of Security, Head of SRE, Head of Customer Success | Head of Finance, Head of Legal |
| Scope-change decisions (above PM authority) | PM (proposes) | Tomislav Hessford (decides) + CTO (concurs) | Affected squad leads, risk owners | All squad leads |
| Cut-over communications to enterprise customers | Quentin Ostrowski | Linnaea Korhonen | PM, SRE Head | Sponsor, CTO |

## 3. Standing Authority Reminders

The PM has standing authority for (`01_project_charter.md` section 10):

- Within-squad work re-prioritisation
- Change requests with cumulative budget impact under $20,000
- Convening cross-squad sync meetings

The PM escalates anything above these thresholds to the Accountable named in each row of the workstream table.

---

**References:** `01_project_charter.md`, `02_scope_statement.md`, `06_stakeholder_register.md`, `09_change_log.md`.
