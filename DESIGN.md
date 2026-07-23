# Terminal Design System
## 1. Purpose
Cogtrix is a terminal text-user-interface application. This file is the visual and interaction template for terminal-facing surfaces only.

Use it for:
- prompt wording
- slash-command UX
- streamed assistant output
- progress indicators
- status lines
- tables and lists
- confirmation prompts
- setup and configuration flows
- error, warning, and success messaging

Do not treat this as a web design system. It is for ANSI/Rich terminal rendering, with a no-color fallback.

## 2. Design Goals
- Precision over decoration: the UI should feel engineered, calm, and dense without becoming noisy.
- Dark-terminal native: assume a dark terminal background first, but degrade cleanly when colors are unavailable.
- Hierarchy through contrast and light single-line framing.
- Fast scanning: every screen should reveal status, next action, and risk quickly.
- Low visual fatigue: avoid harsh white blocks, color overload, gradients, or excessive decoration.
- Retro-minimalist soul with modern clarity: Inspired by 1990s DOS classics (Borland Turbo Vision, Norton Commander, Central Point Anti-Virus) for structure, framing, and hotkeys, while taking clean scannability from modern terminal tools like btop++.

## 3. Visual Language
### Base Tone
- Default screen assumption: near-black terminal background
- Primary foreground: soft near-white (`#f7f8f8`)
- Surfaces created through consistent light single-line box drawing, generous spacing, and subtle indentation

### Accent Strategy
- One cool cyan accent family for primary actions, active states, commands, and highlighted key letters
- One green family for success and healthy status
- One amber family for caution
- One red family for errors and destructive actions
- Gray scale for secondary hierarchy

### Character
- Crisp, technical, minimal — with a subtle 1990s DOS heartbeat and modern clarity
- Light single-line box drawing (`┌ ┐ └ ┘ ─ │`) used consistently for all panels, sections, and dialogs
- No shadows, no background shading, no gradients, no inverse blocks
- Slightly premium, never playful
- Compact but not cramped

## 4. Color Tokens
### Core
- `fg.primary`: `#f7f8f8`
- `fg.secondary`: `#d0d6e0`
- `fg.muted`: `#8a8f98`
- `fg.subtle`: `#62666d`

### Accent
- `accent.primary`: `#00d7ff` (signature cyan)
- `accent.strong`: `#00a5cc`
- `accent.hover`: `#40e0ff`

### Status
- `success`: `#10b981`
- `success.strong`: `#27a644`
- `warning`: `#f59e0b`
- `danger`: `#ef4444`
- `info`: `#60a5fa`

### Lines
- `line.subtle`: `rgba(255,255,255,0.05)`
- `line.standard`: `rgba(255,255,255,0.08)`

## 5. Rich Style Mapping
- Primary text: `white`
- Secondary text: `bright_white`
- Muted metadata: `bright_black`
- Accent text / commands / highlighted key letters: `bold cyan`
- Success: `bold green`
- Warning: `bold yellow`
- Error: `bold red`
- Paths and identifiers: `#d0d6e0`
- Dim annotations: `dim`
- Checkmarks (✓): `bold green`

## 6. Typography and Emphasis
Hierarchy comes from:
- Light single-line box-drawing frames
- line breaks
- indentation
- selective bold
- dimming
- highlighted key letters (shortcut letter in bold cyan)

### Hotkey Highlighting
- In menus, confirmations, and option lists, the accelerator letter is highlighted in bold cyan.
- Format: `<accent><b>Y</b></accent>es` or `<accent><u>V</u></accent>erify`

## 7. Layout Rules
### Width
- Primary target: 80 columns (the sacred DOS sweet spot)

### Persistent Context Bars
- **Top bar** (always visible): Application name on the left (normal text, no inverse block), current mode/context on the right
- **Bottom bar** (always visible): Prompt on the left, stats or short help on the right

### Rhythm & Alignment
- Use blank lines to separate major blocks
- Left-align almost everything
- Right-align only compact metadata

### Indentation
- Use 2 spaces for nested details

### Multi-Panel Use
- Default to a single focused content panel
- Use multiple framed panels sparingly and only when they genuinely improve comparison or workflow

## 8. Structural Patterns
### Standard Screen Structure
1. Top bar
2. Primary framed content area (single panel by default)
3. Optional secondary panels (used sparingly)
4. Bottom bar with prompt

## 9. Component Patterns
### Top Bar (example — no inverse block)
```
Cogtrix ────────────────────────────────────────────────────── Tools
```

### Bottom Bar + Prompt
```
───────────────────────────────────────────────────── 1.2s ↑ 842 ↓ 214
cogtrix ›
```

### Section / Panel Header
```
┌─ Available Tools ─────────────────────────────────────────────┐
```

### Confirmation Box
```
┌─ Confirmation ────────────────────────────────────────────────────────────┐
│ Run `uv sync` in the project environment?                                 │
│ This may install or update dependencies.                                  │
│                                                                           │
│ Choices: <accent>Y</accent>es / <accent>N</accent>o   (Y is safe default) │
└───────────────────────────────────────────────────────────────────────────┘
```

### Option List with Highlighted Keys & Checkmarks
```
┌─ Verify Integrity ─────────────────────────────────────────────┐
│ ✓ <accent>C</accent>reate New Checksums                        │
│   <accent>C</accent>reate Checksums on Floppy                  │
│   <accent>D</accent>isable Alarm Sound                         │
│ ✓ <accent>C</accent>reate Infection Report                     │
│ ✓ <accent>P</accent>rompt while Detect                         │
│                                                                │
│ Press highlighted letter or <accent>Enter</accent> to continue │
└────────────────────────────────────────────────────────────────┘
```

### Status Summary
```
┌─ Validation ───────────────────────────────────────────────────┐
│ [ok]   ruff                                                    │
│ [ok]   black --check                                           │
│ [warn] pyright not run                                         │
└────────────────────────────────────────────────────────────────┘
```

## 10. Streaming and Progress
### Streaming Output
- Assistant text remains the focal point
- Streaming indicators are subtle

### Spinner / Progress
- Compact spinner + short action phrase
- Example: `▶ Thinking…` or `▶ Running ruff…`

### Final Stats
- Placed in bottom bar, right-aligned, dimmed
- Format: `1.2s ↑ 842 ↓ 214`

## 11. Wording Rules
- Be concise and direct
- State the next action clearly
- Do not use cheerful filler
- Error messages should say what failed and what to do next
- Confirmation prompts should identify risk plainly

## 12. State-Specific Guidance
### Success
- Short and calm
- Example: `Configuration saved.`

### Warning
- State the risk, then the consequence

### Error
- Start with the failed action + actionable cause when known

### Empty State
- State absence, then next step
- Example: `No pinned tools loaded. Use /tools load <name>.`

## 13. Accessibility and Fallbacks
- Every colored signal must also have a text label or glyph
- Respect `NO_COLOR`
- The interface must remain fully usable in monochrome terminals
- Fallback box drawing to plain ASCII (`+ - |`)
- Fallback spinner to simple `-\|/`

## 14. Do and Do Not
### Do
- Use single-line box drawing consistently for all frames
- Keep top and bottom bars present for the application feel
- Highlight hotkeys in cyan for instant keyboard discoverability
- Prefer spacing and light frames
- Keep output scannable at 80 columns

### Do Not
- Do not use shadows or inverse blocks
- Do not mix single-line and double-line boxes
- Do not overuse frames or color
- Do not remove top and bottom bars
- Do not hide meaning in color alone
- Do not add gradients or background shading

## 15. Example Patterns
### Full Screen Structure
```
Cogtrix ────────────────────────────────────────────────────── Tools
┌─ Available Tools ─────────────────────────────────────────────┐
│ ✓ <accent>W</accent>eb Search                                 │
│ ✓ <accent>S</accent>hell                                      │
│   <accent>F</accent>ile Write                                 │
│   <accent>M</accent>odel Switch                               │
└───────────────────────────────────────────────────────────────┘

───────────────────────────────────────────────────────────── Ready
cogtrix ›
```

### Confirmation Box
```
┌─ Confirmation ────────────────────────────────────────────────────────────────┐
│ Run `uv sync` in the project environment?                                     │
│ This may install or update dependencies.                                      │
│                                                                               │
│ Choices: <accent><b><u>Y</u></b></accent>es / <accent><b>N</b></accent>o      │
└───────────────────────────────────────────────────────────────────────────────┘
```

## 16. Implementation Notes
- Prefer `rich` styles over custom escape-sequence logic
- Keep shared style choices centralized
- Use semantic names for style helpers
- Test at exactly 80 columns
- Test both color and no-color output paths
- Ensure all frames use single-line box drawing consistently

---
