# Stakeholder Register — Project Nimbus

**Document ID:** NIMB-STKH-001
**Version:** 1.2
**Last updated:** 2026-07-09 (post-Q2 steering)

14 stakeholders are tracked. Each entry records the stakeholder's role, organisational position, influence and interest levels (Low / Medium / High), engagement strategy, and communication preferences.

The 2 × 2 influence/interest grid drives the communication plan in `18_communication_plan.md`.

---

## 1. Sponsor and Senior Decision Authorities

### Tomislav Hessford — Chief Operating Officer

- **Role on Nimbus:** Programme Sponsor (signs SC-1 through SC-5; approves budget envelope changes > $20,000; holds the contingency reserve)
- **Influence:** High
- **Interest:** High
- **Engagement strategy:** Weekly written status (Monday 09:00 UTC) + monthly 30-minute sync (last Friday). Escalations called immediately for High-High risks (currently R-12 + R-19).
- **Preferred communication:** Written first, voice if blocked.

### Avantika Sundararaman — Chief Technology Officer

- **Role on Nimbus:** Senior decision authority on technical scope; co-signs all in-flight scope changes; holds engineering bandwidth allocation for D-09.
- **Influence:** High
- **Interest:** High
- **Engagement strategy:** Same cadence as the Sponsor. Critical-path technical decisions go through her before the steering committee sees them.
- **Preferred communication:** Slack (DM), then written summary.

### Pernille Vrieze — Head of Security

- **Role on Nimbus:** Security review sign-off on D-01, D-03, D-04, D-05, D-10. Risk owner for any security-classified risk (none currently).
- **Influence:** High
- **Interest:** Medium
- **Engagement strategy:** Monthly 30-minute review; per-deliverable security sign-off in writing.

### Bartholomew Okafor-Sing — Head of SRE

- **Role on Nimbus:** Operational sign-off on D-02, D-06, D-08, D-10. Risk owner for R-17 (observability gap).
- **Influence:** High
- **Interest:** High
- **Engagement strategy:** Weekly 20-minute sync (Tuesday); attends every cut-over rehearsal.

### Linnaea Korhonen — Head of Customer Success

- **Role on Nimbus:** Customer-impact sign-off; risk owner for R-19 (Helmsdale acceptance window); manages enterprise communication.
- **Influence:** Medium
- **Interest:** High
- **Engagement strategy:** Weekly written digest + ad-hoc syncs as cut-over approaches.

---

## 2. Squad Leads

### Aldous Pemberton-Riggs — Platform Squad Lead

- **Influence:** Medium
- **Interest:** High
- **Engagement strategy:** Daily stand-up with the PM (15 min, 09:30 UTC).

### Beatriz Cazadora-Olesen — Data Squad Lead

- **Influence:** Medium
- **Interest:** High
- **Engagement strategy:** Daily stand-up. Currently the busiest of the squad leads — owns R-12, R-15, and the critical-path milestone M-DB-R.

### Hyeon-Jin Park — Migration Squad Lead

- **Influence:** Medium
- **Interest:** High
- **Engagement strategy:** Daily stand-up.

### Vukašin Andrássy — Networking Squad Lead

- **Influence:** Medium
- **Interest:** High
- **Engagement strategy:** Daily stand-up.

---

## 3. Vendor

### Yusuf Almasi — AcmeCloud Technical Account Manager

- **Role on Nimbus:** AcmeCloud's single point of contact for capacity, SLA, and escalation. Holds R-12 mitigation and R-13 conversion-to-firm-commitment.
- **Influence:** Medium (high on AcmeCloud-side, no direct authority here)
- **Interest:** High
- **Engagement strategy:** Weekly 30-minute sync (Thursday). Replaces the previous TAM who rotated off the account in mid-March; this rotation is the proximate cause of the M-LZ slip (see M1 status, `10_status_report_m1.md`).

---

## 4. Cross-Functional Stakeholders

### Eberhard Lindqvist-Marais — Head of Finance

- **Role on Nimbus:** Owns the budget tracking line in `07_budget.md`. Approves contingency call-downs.
- **Influence:** Medium
- **Interest:** Medium
- **Engagement strategy:** Monthly written report + ad-hoc when budget variance crosses ±3%.

### Asha Wickremasinghe — Head of Legal

- **Role on Nimbus:** Reviews the AcmeCloud contract amendments; signs off on data-residency commitments to enterprise customers; on standby for the Helmsdale acceptance-window escalation (R-19).
- **Influence:** Medium
- **Interest:** Low (unless a contract issue surfaces)
- **Engagement strategy:** Notified per significant decision; otherwise quarterly summary.

### Marcus Aurelius Babatunde — Head of Engineering

- **Role on Nimbus:** Provides engineering capacity for D-09; arbitrates between Nimbus and Q3 product roadmap for the engineers identified in R-16.
- **Influence:** Medium
- **Interest:** Medium
- **Engagement strategy:** Bi-weekly 20-minute sync.

### Sebastián Roiglund-Faye — Compliance Officer

- **Role on Nimbus:** Validates data-residency controls before Helmsdale validation begins.
- **Influence:** Medium
- **Interest:** Medium
- **Engagement strategy:** Pre-validation review (scheduled 2026-10-29).

### Quentin Ostrowski — Communications Lead

- **Role on Nimbus:** Coordinates customer-facing communication around the cut-over maintenance window.
- **Influence:** Low
- **Interest:** Medium
- **Engagement strategy:** Engaged from 2026-09-15 (8 weeks pre-cut-over).

---

## 5. Influence / Interest Grid

| | Low Interest | Medium Interest | High Interest |
|---|---|---|---|
| **High Influence** | — | Pernille Vrieze | Tomislav Hessford, Avantika Sundararaman, Bartholomew Okafor-Sing |
| **Medium Influence** | Asha Wickremasinghe | Eberhard Lindqvist-Marais, Marcus Aurelius Babatunde, Sebastián Roiglund-Faye | Linnaea Korhonen, Aldous Pemberton-Riggs, Beatriz Cazadora-Olesen, Hyeon-Jin Park, Vukašin Andrássy, Yusuf Almasi |
| **Low Influence** | — | Quentin Ostrowski | — |

**Counts:** 14 stakeholders total. 4 are High-Influence; 9 are High-Interest. The full quadrant of "High Influence × High Interest" (the "manage closely" quadrant) has 3 stakeholders: the Sponsor, the CTO, and the Head of SRE. The communication plan in `18_communication_plan.md` budgets the most PM time on this quadrant.

---

**References:** `01_project_charter.md`, `05_risk_register.md`, `08_raci_matrix.md`, `18_communication_plan.md`.
