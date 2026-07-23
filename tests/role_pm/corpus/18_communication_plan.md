# Communication Plan — Project Nimbus

**Document ID:** NIMB-COMM-001
**Version:** 1.2
**Last updated:** 2026-07-12

Defines the cadence, format, and audience for every Nimbus communication. Drives PM time-allocation against the influence/interest grid in `06_stakeholder_register.md`.

---

## 1. Cadence Map

| Cadence | Audience | Format | Owner |
|---|---|---|---|
| Daily 09:30 UTC | Squad leads (A.P-R, B.C-O, H-J.P, V.A) + PM | 15-min stand-up, video | PM |
| Weekly Monday 09:00 UTC | Sponsor (Tomislav Hessford), CTO (Avantika Sundararaman), Head of SRE (Bartholomew Okafor-Sing) | Written status digest (Slack + email mirror) | PM |
| Weekly Tuesday 14:00 UTC | Head of SRE + PM | 20-min sync, video | PM |
| Weekly Thursday 15:00 UTC | AcmeCloud TAM (Yusuf Almasi) + PM | 30-min sync, video | PM |
| Bi-weekly | Head of Engineering (Marcus Aurelius Babatunde) + PM | 20-min sync, video | PM |
| Monthly last Friday | Sponsor + PM | 30-min sync, video | Sponsor |
| Monthly first working day | All stakeholders | Written status report (this is the M1/M2/M3/M4 series in `10` through `13`) | PM |
| Quarterly | Steering committee | 90-min meeting + written minutes | Sponsor (chair), PM (minutes) |
| Per significant decision | Affected parties | Written decision brief (logged in `14_decision_log.md`) | Decision owner |
| Per material risk update | Risk owner + Sponsor (for High-High) | Slack DM + written follow-up | Risk owner |
| T-21 days from cut-over | Enterprise customers | Email (Customer Success template, approved 2026-04-30) | Quentin Ostrowski |

## 2. Status-Report Discipline

Every monthly status report uses the format defined in the system prompt's Status Reporting Format section. Each report:

- Opens with the Yellow/Green/Red headline.
- Carries forward last month's open actions.
- Surfaces every new risk and every status change to existing risks.
- Documents budget variance to the dollar.
- Names the WBS items that moved this month.

## 3. Escalation Communication

- **High-High risks (R-12, R-19):** Sponsor notified immediately on any material status change.
- **Budget variance > ±3%:** Head of Finance + Sponsor notified within 1 working day.
- **Schedule slip > 1 week on a critical-path milestone:** Sponsor + CTO notified within 1 working day; covered at next steering meeting.
- **AcmeCloud SLA breach:** Sponsor + Head of SRE notified within 1 hour.

## 4. PM Time Allocation (Target)

The plan budgets PM time against the influence/interest grid (`06_stakeholder_register.md` section 5):

- **Manage closely (High Influence × High Interest — 3 stakeholders):** ~40% of PM time.
- **Keep satisfied (High Influence × Lower Interest — 1 stakeholder, Pernille Vrieze):** ~10%.
- **Keep informed (Lower Influence × High Interest — 9 stakeholders):** ~35%.
- **Monitor (Lower Influence × Lower Interest):** ~5%.
- **External / vendor:** ~10%.

This is a guideline, not a contract.

## 5. Communication Channel Norms

- **Slack DM:** for time-sensitive single-recipient notes. Always followed by a written summary if the topic is decision-relevant.
- **Slack channel (#nimbus-programme):** for squad-level coordination. Squad leads + PM are the active participants; everyone else is read-only.
- **Email:** for formal written communication to stakeholders outside the daily channel set (Finance, Legal, Communications).
- **Video meetings:** for any conversation that needs reading nonverbal signal (steering meetings, escalations, customer conversations).
- **No status by emoji:** project-status flags (Green/Yellow/Red) only travel in written reports, never in a passing Slack comment.

---

**References:** `01_project_charter.md`, `06_stakeholder_register.md`.
