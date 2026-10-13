"""CSS for the ResearchDesk look (the "Refined dark" design).

Colours, fonts, corner radius and heading sizes come from
.streamlit/config.toml; this file adds what a theme cannot express: the
sidebar navigation, cards, chips, steppers, bars and the login screen.
The CSS variables below must stay in step with that config (accent #3DD6B5,
background #0E1116).

Selectors use Streamlit's data-testid attributes and the st-key-<key> class
Streamlit adds to widgets that have a key= (checked against Streamlit 1.65).
If a future Streamlit renames one, that rule simply stops applying -- nothing
breaks, that element just falls back to the plain theme.
"""

GLOBAL_CSS = r"""
:root {
  --rd-bg: #0E1116;
  --rd-side: #12161D;
  --rd-card: #151A23;
  --rd-active: #1C2330;
  --rd-line: #262D3A;
  --rd-side-line: #232A36;
  --rd-text: #E8EBF2;
  --rd-text2: #C6CDDA;
  --rd-muted: #8590A6;
  --rd-muted2: #A3ADBF;
  --rd-accent: #3DD6B5;
  --rd-ink: #04221C;
  --rd-ok: #5FD3A0;
  --rd-warn: #F2B84B;
  --rd-bad: #F2777A;
  --rd-mono: "DM Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}

/* ---- page frame ---- */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"],
.block-container {
  max-width: 1180px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-top: 2rem;
  padding-bottom: 3rem;
}

/* ---- top bar: page title + page controls ---- */
.st-key-topbar {
  border-bottom: 1px solid var(--rd-side-line);
  padding-bottom: 14px;
  margin-bottom: 22px;
}
.rd-topbar-title { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.rd-title { font-size: 1.2rem; font-weight: 600; letter-spacing: -0.01em; }
.rd-subtitle { color: var(--rd-muted); font-size: 0.93rem; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] { background: var(--rd-side); }
/* margin-bottom, not padding: Streamlit's sidebar vertical block spaces its
   children with CSS `gap`, which only responds to margin on a child, not
   that child's own internal padding -- padding-bottom here would be
   invisible no matter how large, because it never reaches the next sibling. */
.rd-brand-row { display: flex; align-items: center; gap: 10px; padding: 4px 6px 0; margin-bottom: 18px; }
.rd-logo {
  flex: 0 0 34px;
  width: 34px;
  height: 34px;
  border-radius: 10px;
  background: var(--rd-accent);
  color: var(--rd-ink);
  display: flex;
  align-items: center;
  justify-content: center;
}
.rd-brand { font-size: 1.13rem; font-weight: 700; letter-spacing: -0.01em; line-height: 1.2; }
.rd-tagline { font-size: 0.8rem; color: var(--rd-muted); }
.rd-section {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--rd-muted);
  padding: 0 10px 4px;
  margin-top: 6px;
}

/* navigation and chat list: plain rows, not boxed buttons */
[class*="st-key-nav_"] button,
[class*="st-key-chat_"] button {
  width: 100%;
  justify-content: flex-start;
  text-align: left;
  border: 0;
  background: transparent;
  color: var(--rd-text2);
  font-weight: 500;
  border-radius: 10px;
  min-height: 40px;
  padding: 0 10px;
  gap: 12px;
}
[class*="st-key-chat_"] button { min-height: 36px; font-weight: 400; color: var(--rd-muted2); }
[class*="st-key-nav_"] button:hover,
[class*="st-key-chat_"] button:hover { background: rgba(255, 255, 255, 0.04); }
[class*="st-key-chat_"] button p {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
}
[class*="st-key-nav_active_"] button { background: var(--rd-active) !important; color: #FFFFFF; }
[class*="st-key-nav_active_"] [data-testid="stIconMaterial"] { color: var(--rd-accent); }
[class*="st-key-chat_active_"] button { background: var(--rd-active) !important; color: var(--rd-text); }

.st-key-new_chat button { min-height: 44px; border-radius: 12px; font-weight: 600; }

[class*="st-key-nav_"] button {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  text-align: left;
}
/* `justify-content` only has any effect on an element that is itself a flex
   (or grid) container -- setting it on a nested wrapper without also
   setting `display: flex` on that same wrapper is a no-op, which is why
   the single narrower rule this replaced had no visible effect. Streamlit's
   own button styling centers icon+label by default (ordinary button
   behaviour), and does so somewhere inside the button's descendant chain
   (exact depth not confirmed without browser devtools) -- so every
   descendant level is forced to flex+left here, not just one, with
   `!important` since Streamlit's own rule is also a default button style
   that must be beaten regardless of which selector it actually uses. */
[class*="st-key-nav_"] button * {
  justify-content: flex-start !important;
  text-align: left !important;
}
/* Recent-chat row buttons: confirmed via real rendered DOM that they share
   the exact same BaseWeb wrapper structure as nav buttons (identical
   div/span class names in outerHTML), and the same wrapper levels compute
   justify-content: center without this override. text-align was already
   correct ("left") at every level in that inspection, so only
   justify-content needs overriding here -- the nav fix's `> div, p {
   display: flex }` rule isn't needed either, since chat buttons have no
   icon and so no icon+label pair to arrange side by side at that level. */
[class*="st-key-chat_"] button * {
  justify-content: flex-start !important;
}
[class*="st-key-nav_"] button > div,
[class*="st-key-nav_"] button p {
  display: flex !important;
  align-items: center !important;
  width: 100%;
}
[class*="st-key-nav_"] button p {
  margin: 0;
  line-height: 1.35;
}
[class*="st-key-nav_"] button [data-testid="stIconMaterial"] {
  flex-shrink: 0;
}

/* primary buttons: dark text on the light accent */
button[kind="primary"],
[data-testid="stBaseButton-primary"] { color: var(--rd-ink); font-weight: 600; }
button[kind="primary"] *,
[data-testid="stBaseButton-primary"] * { color: var(--rd-ink); }

/* signed-in user at the bottom of the sidebar */
[data-testid="stSidebarUserContent"] > [data-testid="stVerticalBlock"] { min-height: calc(100vh - 6rem); }
.st-key-user_chip {
  margin-top: auto;
  border-top: 1px solid var(--rd-side-line);
  padding-top: 14px;
}
.rd-user-chip { display: flex; align-items: center; gap: 10px; }
.rd-avatar {
  flex: 0 0 32px;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: #232B3A;
  color: var(--rd-text2);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.85rem;
  font-weight: 600;
}
.rd-user-text { min-width: 0; }
.rd-k { font-size: 0.78rem; color: var(--rd-muted); }
.rd-email { font-size: 0.85rem; color: var(--rd-text2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ---- chips, pills, ids ---- */
.rd-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.6rem;
  margin-top: 0.35rem;
  font-size: 0.78rem;
  opacity: 0.85;
}
.rd-pill {
  border: 1px solid rgba(128, 128, 128, 0.35);
  border-radius: 999px;
  padding: 0.05rem 0.65rem;
  white-space: nowrap;
}
.rd-trace { font-family: var(--rd-mono); }
.rd-wrap { word-break: break-all; }
.rd-note { color: var(--rd-muted); font-size: 0.82rem; }

/* ---- chat content (also used on the Agent Trace page) ---- */
.rd-cite {
  display: inline-block;
  margin: 0 1px 0 4px;
  padding: 0 6px;
  border-radius: 6px;
  background: rgba(61, 214, 181, 0.16);
  color: var(--rd-accent);
  font-size: 0.72rem;
  font-weight: 600;
  vertical-align: 2px;
}
.rd-sources { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.rd-source {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 30px;
  padding: 0 12px;
  border: 1px solid var(--rd-line);
  border-radius: 10px;
  background: var(--rd-side);
  color: var(--rd-text2);
  font-size: 0.82rem;
}
.rd-source b { color: var(--rd-accent); font-weight: 600; }

/* ---- cards and summaries ---- */
.rd-card {
  background: var(--rd-card);
  border: 1px solid var(--rd-line);
  border-radius: 14px;
  padding: 18px;
  margin-bottom: 4px;
}
.rd-summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 20px; }
.rd-field .rd-v { font-size: 0.97rem; word-break: break-word; }
.rd-pipeline { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 10px; }
.rd-step {
  border: 1px solid var(--rd-line);
  border-radius: 999px;
  padding: 3px 11px;
  font-size: 0.8rem;
  color: var(--rd-muted2);
  white-space: nowrap;
}
.rd-step-strong { background: rgba(61, 214, 181, 0.16); color: var(--rd-accent); border-color: transparent; font-weight: 600; }
.rd-arrow { color: var(--rd-muted); }
.rd-queries { margin-top: 14px; }
.rd-section-title { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin: 22px 0 10px; }
.rd-section-title h3 { margin: 0; padding: 0; font-size: 1rem; font-weight: 600; }
.rd-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 16px 0; }
.rd-tile { padding: 16px; }
.rd-tile-v { font-size: 1.15rem; font-weight: 600; }
.rd-validation { display: inline-flex; align-items: center; gap: 8px; font-weight: 600; }

/* ---- result rows (retrieved passages, evidence, failures) ---- */
.rd-result {
  border: 1px solid var(--rd-line);
  border-radius: 12px;
  background: var(--rd-card);
  margin-bottom: 10px;
}
.rd-result summary,
.rd-result-static {
  display: flex;
  align-items: center;
  gap: 14px;
  min-height: 54px;
  padding: 0 16px;
  cursor: pointer;
  list-style: none;
}
.rd-result-static { cursor: default; }
.rd-result summary::-webkit-details-marker { display: none; }
.rd-rank {
  flex: 0 0 26px;
  height: 26px;
  border-radius: 50%;
  background: var(--rd-active);
  color: var(--rd-text2);
  font-size: 0.8rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
}
.rd-result-name { flex: 1; min-width: 0; }
.rd-fname { font-size: 0.95rem; font-weight: 500; }
.rd-loc { color: var(--rd-muted); font-size: 0.85rem; }
.rd-badges { display: flex; gap: 8px; flex-wrap: wrap; }
.rd-badges-wrap { display: flex; gap: 8px; flex-wrap: wrap; }
.rd-badge {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 8px;
  border: 1px solid var(--rd-line);
  color: var(--rd-muted2);
  font-size: 0.78rem;
  font-family: var(--rd-mono);
}
.rd-badge-accent { background: rgba(61, 214, 181, 0.16); color: var(--rd-accent); border-color: transparent; font-weight: 600; }
.rd-badge-tool { color: var(--rd-text2); }
.rd-chevron { color: var(--rd-muted); display: flex; transition: transform 0.15s; }
details[open] > summary .rd-chevron { transform: rotate(180deg); }
.rd-result-body { padding: 0 16px 16px 56px; }
.rd-passage {
  margin: 6px 0 0;
  padding: 12px 14px;
  background: var(--rd-bg);
  border: 1px solid var(--rd-line);
  border-radius: 10px;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 260px;
  overflow: auto;
  font-family: inherit;
  font-size: 0.88rem;
  line-height: 1.6;
  color: var(--rd-text2);
}
/* "Tools used": a plain row, so its content is not indented like a passage */
.rd-tools summary { min-height: 48px; }
.rd-tools .rd-result-body { padding-left: 16px; }
.rd-fail-q { color: var(--rd-muted2); font-size: 0.9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ---- agent trace stepper ---- */
.rd-stepper { padding: 20px; }
.rd-trace-step { display: flex; gap: 16px; }
.rd-trace-rail { display: flex; flex-direction: column; align-items: center; }
.rd-step-num {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background: rgba(61, 214, 181, 0.16);
  color: var(--rd-accent);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.83rem;
  font-weight: 700;
}
.rd-rail-line { width: 2px; flex: 1; min-height: 22px; background: var(--rd-line); margin-top: 6px; }
.rd-trace-step:last-child .rd-rail-line { display: none; }
.rd-trace-main { flex: 1; min-width: 0; padding-bottom: 18px; }
.rd-trace-step:last-child .rd-trace-main { padding-bottom: 0; }
.rd-trace-head { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; min-height: 30px; }
.rd-trace-title { flex: 1 1 220px; min-width: 0; }
.rd-step-name { font-size: 0.95rem; font-weight: 600; }
.rd-step-desc { font-size: 0.82rem; color: var(--rd-muted); }
.rd-trace-tags { display: flex; gap: 8px; flex-wrap: wrap; }
.rd-duration { font-family: var(--rd-mono); font-size: 0.8rem; color: var(--rd-muted2); }
.rd-status { display: inline-flex; align-items: center; gap: 6px; font-size: 0.8rem; }
.rd-st-ok { color: var(--rd-ok); }
.rd-st-warn { color: var(--rd-warn); }
.rd-st-bad { color: var(--rd-bad); }

/* ---- status dots ---- */
.rd-status-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; margin-bottom: 0.4rem; }
.rd-dot { width: 8px; height: 8px; min-width: 8px; border-radius: 50%; display: inline-block; }
.rd-dot-ok { background-color: var(--rd-ok); }
.rd-dot-warn { background-color: #b45309; }
.rd-dot-bad { background-color: var(--rd-bad); }

/* ---- documents ---- */
.rd-filebadge {
  width: 42px;
  height: 42px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.04em;
}
.rd-ft-pdf { background: rgba(242, 119, 122, 0.14); color: #F2777A; }
.rd-ft-txt { background: rgba(143, 180, 255, 0.14); color: #8FB4FF; }
.rd-ft-md { background: rgba(190, 160, 255, 0.14); color: #BEA0FF; }
.rd-ft-other { background: var(--rd-active); color: var(--rd-muted2); }
.rd-doc-name {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 10px;
  line-height: 1.4;
  word-break: break-word;
}
/* `st.container(border=True, key=...)` renders the border on an inner
   wrapper one level below the element the st-key-<key> class is actually
   attached to -- padding on the outer element alone sits outside that
   border and has no visible effect, which is why the single rule this
   replaced didn't change anything. Padding is set at both levels so it
   applies wherever the real bordered box turns out to be. */
[class*="st-key-doc_"],
[class*="st-key-doc_"] > div {
  padding: 18px;
}
[class*="st-key-doc_"] .rd-meta {
  gap: 8px;
  row-gap: 8px;
  margin-top: 0.4rem;
}
[class*="st-key-doc_"] .rd-meta .rd-pill,
[class*="st-key-doc_"] .rd-meta .rd-trace {
  max-width: 100%;
}
.rd-date { color: var(--rd-muted); font-size: 0.82rem; white-space: nowrap; text-align: right; }
/* Delete button on a document card: confirmed via real rendered DOM that
   its column (date_col/delete_col = action_col.columns([2, 1])) measured
   only ~62px wide -- too narrow for an icon + "Delete" at `white-space:
   normal` (the actual computed value with nothing overriding it), which
   wraps the word across two lines. The column ratio is widened alongside
   this rule (see action_col.columns([3, 2]) in streamlit_app.py) so there
   is enough room normally; this is kept as a guarantee that even if the
   card is narrower than typical, the label still never wraps. */
[class*="st-key-delete_doc_"] button {
  white-space: nowrap;
}
[class*="st-key-delete_doc_"] button p {
  white-space: nowrap !important;
}
[data-testid="stFileUploaderDropzone"] {
  border: 1.5px dashed #2F384A;
  border-radius: 14px;
  background: var(--rd-side);
  padding: 20px 22px;
}

/* ---- evaluation ---- */
.rd-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin-bottom: 16px; }
.rd-kpi { padding: 16px; }
.rd-kpi-v { font-size: 1.6rem; font-weight: 600; letter-spacing: -0.01em; line-height: 1.3; }
.rd-kpi-sub { font-size: 0.8rem; color: var(--rd-muted); }
.rd-card-title { margin: 0 0 12px; padding: 0; font-size: 1rem; font-weight: 600; }
.rd-card-note { margin: 0 0 14px; font-size: 0.82rem; color: var(--rd-muted); }
.rd-kvs .rd-kv {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 9px 0;
  border-bottom: 1px solid var(--rd-line);
  font-size: 0.92rem;
}
.rd-kvs .rd-kv:last-child { border-bottom: 0; }
.rd-kv span { color: var(--rd-muted2); }
.rd-bar-block { margin-bottom: 18px; }
.rd-bar-title { font-size: 0.9rem; font-weight: 500; margin-bottom: 8px; }
.rd-bar { display: flex; height: 10px; border-radius: 999px; overflow: hidden; background: var(--rd-active); }
.rd-seg { display: block; height: 100%; }
.rd-seg-0 { background: var(--rd-accent); }
.rd-seg-1 { background: #8FB4FF; }
.rd-seg-2 { background: #F2777A; }
.rd-seg-3 { background: #5B6577; }
.rd-legends { display: flex; flex-wrap: wrap; gap: 6px 16px; margin-top: 8px; }
.rd-legend { display: inline-flex; align-items: center; gap: 6px; font-size: 0.8rem; color: var(--rd-muted2); }
.rd-swatch { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }
.rd-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.rd-chip {
  display: inline-flex;
  gap: 8px;
  align-items: center;
  border: 1px solid var(--rd-line);
  border-radius: 999px;
  padding: 3px 12px;
  font-size: 0.8rem;
  color: var(--rd-muted2);
}
.rd-chip strong { color: var(--rd-text); }
"""

# Chat page only: a readable column, the user's bubbles, and the assistant's card.
CHAT_CSS = r"""
[data-testid="stMainBlockContainer"],
.block-container { max-width: 900px !important; }

.rd-user-row { display: flex; justify-content: flex-end; margin: 6px 0; }
.rd-user {
  max-width: 80%;
  background: var(--rd-active);
  border: 1px solid var(--rd-line);
  padding: 10px 16px;
  border-radius: 16px 16px 4px 16px;
  font-size: 0.97rem;
  line-height: 1.5;
  word-break: break-word;
}

[data-testid="stChatMessage"] { background: transparent; padding: 0.25rem 0; gap: 0.9rem; align-items: flex-start; }
[data-testid="stChatMessageContent"] {
  background: var(--rd-card);
  border: 1px solid var(--rd-line);
  border-radius: 14px;
  padding: 14px 18px;
}
[data-testid="stChatMessageAvatarCustom"] { background: rgba(61, 214, 181, 0.14); border-radius: 9px; }
"""

# Login / sign-up: no sidebar, one centred card, tabs drawn as a segmented switch.
AUTH_CSS = r"""
[data-testid="stMainBlockContainer"],
.block-container { max-width: 460px !important; padding-top: 9vh; }

.rd-auth-brand { display: flex; flex-direction: column; align-items: center; gap: 6px; text-align: center; margin-bottom: 24px; }
.rd-logo-lg { flex: 0 0 48px; width: 48px; height: 48px; border-radius: 14px; }
.rd-auth-name { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2; }
.rd-auth-note { text-align: center; font-size: 0.82rem; color: var(--rd-muted); margin-top: 18px; }

.st-key-auth_card {
  background: var(--rd-card);
  border: 1px solid var(--rd-line);
  border-radius: 16px;
  padding: 24px 24px 12px;
}

[data-testid="stTabs"] [role="tablist"] {
  background: var(--rd-bg);
  border: 1px solid var(--rd-line);
  border-radius: 12px;
  padding: 3px;
  gap: 2px;
}
[data-testid="stTabs"] [role="tablist"] > *:not([data-testid="stTab"]) { display: none; }
[data-testid="stTab"] { flex: 1; justify-content: center; border-radius: 9px; height: 34px; color: var(--rd-muted2); }
[data-testid="stTab"][aria-selected="true"] { background: #263044; color: #FFFFFF; }

[data-testid="stForm"] { border: 0; padding: 0; }
[data-testid="stFormSubmitButton"] button { width: 100%; min-height: 44px; }

/* The "Press Enter to submit form" hint Streamlit renders inside every
   text/password input that lives inside an st.form (confirmed via real
   DOM inspection: a <span> inside [data-testid="InputInstructions"],
   nested directly in the input's own wrapper) sits close enough to the
   field to obscure the email/password text while typing. It is purely
   decorative -- removing it does not touch the <input> element itself, so
   Enter-to-submit, keyboard navigation, and the password show/hide toggle
   (a separate control, not part of this element) are all unaffected.
   Scoped to AUTH_CSS only, so other forms elsewhere in the app (which
   this stylesheet is never injected for) keep showing this hint. */
[data-testid="InputInstructions"] { display: none; }
"""
