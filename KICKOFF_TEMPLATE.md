Starting Stage [N] — [title]. Build me the Claude Code prompt for it.

READ FIRST — open these from project knowledge before writing anything. Do not work from memory
or from what is quoted in this kickoff; the kickoff paraphrases, the files are authoritative.
  1. PROJECT.md — §5 differentiators, §7 definition of done, §8 out of scope. Anything this stage
     would build that §8 excludes is scope creep and must be flagged before the prompt is written.
  2. TECHNICAL_SPEC.md §3 — this stage's definition, and §5 the invariants.
  3. HARDWARE.md — every machine figure the prompt will state. §5.5 the environment fingerprint.
  4. BENCHMARK_PROTOCOL.md — every measurement rule, including §4.1 noise floor and §4.2 warmup.
  5. MEASUREMENTS.md — prior stages' entries, so this stage's prediction is informed by what
     already happened and its gap explanations do not contradict earlier findings.
  6. PERSISTENT.md — all open flags, decisions, constraints, and known traps. §7 is cleared and
     rewritten every session and carries only what the immediately following session needs. §8 is
     NOT cleared: it holds deferred work items owned by a named stage, each with an ID (W1, W2, …),
     the evidence behind it, and what would close it. Read both. §8 is the one that outlives this
     conversation, so an item it names as owned by this stage is a hard obligation and not a note.
  7. CC_PROMPT_FORMAT.md — the house format, before writing the prompt body.
State plainly if any file is missing from project knowledge rather than proceeding without it.

[PERSISTENT §7 — flags relevant to this stage: [name only the flags where THIS stage is affected.
Do NOT add a "standing / likely-surfaces" line, do NOT list flags you ruled out, do NOT add "X is
NOT relevant" exclusions — relevant flags only.]
]

[PERSISTENT §8 — deferred work items. Before writing the prompt, read §8 in full and report to me
by ID: which items name THIS stage as owner and must be resolved or explicitly deferred in this
session; which items this stage's work could inadvertently touch, close, or invalidate even though
it does not own them; and which are irrelevant here. Name the irrelevant ones to me only — do not
carry them into the prompt. See RULE 21 for what the prompt must then do with each.
Confirm §8 exists and is populated. If it does not exist, say so plainly and do not proceed as
though the deferred items are safe in §7 — they are not, because this stage's own session rewrites
that section.]

PREDICTION BRIEFING — before writing the Claude Code prompt, produce the prediction briefing for
this stage per RULE 14. I write my guess in this chat BEFORE you finalize the prompt; my guess
goes into the prompt verbatim so Claude Code can commit it to MEASUREMENTS.md before writing any
code. From Stage 11 onward the prediction comes from the Stage 10 performance model, not from me
— see RULE 14(e).

CC PROMPT BODY FORMAT — before writing the Claude Code prompt, read CC_PROMPT_FORMAT.md in
project knowledge and match its house format (== CAPS == section delimiters, the fixed section
order, NEW FILE N / EDIT FILE N — path + inline asserts, plain build-voice, recon-then-build by
default). See RULE 11.

Give me: (1) the prediction briefing, and the PERSISTENT §8 report required by RULE 21 — owned,
touchable, and irrelevant items by ID; (2) after I supply my guess, the copy-pastable Claude Code
prompt; (3) the checks I run myself at the HARD CHECKPOINT — the RULE 8 user-only set (benchmark
conditions, profiler access, anything needing a device other than the primary). Do NOT hand me
the offline gate: per RULE 10 the agent runs build, unit tests, and the correctness gate itself.
The Claude Code prompt ends with the RULE 10 flow — self-run offline gate -> STOP at green ->
present the user-only checks and WAIT (no commit) -> on my "proceed" run benchmarks and
profiling, fill MEASUREMENTS.md, then the commit sequence; (4) which files I re-add to Claude
project knowledge after the session — at minimum the updated MEASUREMENTS.md, PERSISTENT.md,
LEARNING.md, and the regenerated repo-files.txt.

---

PROMPT CONSTRUCTION RULES (apply before writing any Claude Code prompt):

RULE 1 — SOURCE EVERY NUMBER, PATH, AND CONSTRAINT FROM FOUNDATION DOCS FIRST.
Read TECHNICAL_SPEC.md §3 for the stage definition, HARDWARE.md for any machine figure, and
BENCHMARK_PROTOCOL.md for any measurement rule, before writing them into the prompt. Never from
memory. If two sources disagree, flag it and defer to the real artifact (the code, the config
file, the measured value) over the document.

RULE 1a — SCOPE CHECK AGAINST PROJECT.md §8 BEFORE WRITING THE PROMPT.
Every output the prompt would produce is checked against the out-of-scope list. If the stage as
conceived would produce something §8 excludes, flag it to me before writing the prompt rather
than quietly building it. Also check the stage's outputs against PROJECT.md §7 — a stage that
does not advance a definition-of-done item needs a stated reason for existing.

RULE 2 — NEVER STATE A HARDWARE NUMBER FROM MEMORY.
SM counts, clocks, cache sizes, bandwidth, vector widths, peak throughput: transcribed from
HARDWARE.md or marked explicitly as to-be-measured. Spec-sheet theoretical peaks never enter a
prediction or the performance model.

RULE 3 — PREDICTION GATE. No build session runs without a committed prediction.
Stages 4–9 and 11 carry my prediction verbatim; the prompt instructs Claude Code to write it into
MEASUREMENTS.md and commit BEFORE any implementation code. Exempt: Stage 0, 1, 3, 12 and the
optional stages. Stage 2 predicts the baseline itself. Stage 10 predicts nothing — its result is
an error distribution.

RULE 4 — COUNT FILES AND ASSERTS BEFORE STATING A COUNT. Enumerate, then count.

RULE 5 — INSTRUCTION SCOPE: THE PROMPT TELLS CLAUDE CODE WHAT TO BUILD.
Never instruct Claude Code to derive the stage contract from a document. Doc-reading is mine when
building the prompt; the prompt delivers the output of that reading. Exception: live-repo recon,
which is what DETERMINE sections are for.

RULE 6 — OBJECTION-FIRST ON EVERY PROMPT BEFORE DELIVERING IT.
State the top 2 ways the prompt could produce a wrong result, verify each against project files,
confirm or correct. My pre-delivery check only — never inside the CC prompt.

RULE 7 — SELF-AUDIT BEFORE DELIVERY. Run silently:

  □ Every file path — exists, or is the correct new path?
  □ Scope checked against PROJECT.md §8 — nothing excluded is being built?
  □ Every hardware figure — transcribed from HARDWARE.md, not memory?
  □ Every measurement rule — matches BENCHMARK_PROTOCOL.md?
  □ Prediction block present and verbatim (or stage correctly exempt per RULE 3)?
  □ Prefill and decode separated everywhere a timing appears?
  □ Correctness gate wired into the stage's accept criteria?
  □ Nsight counter collection specified for any GPU stage (Stage 7 onward)?
  □ No sm_80+ intrinsic proposed anywhere (cp.async, ldmatrix, mma)?
  □ No performance figure asserted that has not been measured?
  □ Every count derived by enumerating?
  □ OBJECTION-FIRST done?
  □ House format matches CC_PROMPT_FORMAT.md?

If any box cannot be checked, fix before delivering. Do not tell me you ran the checklist.

RULE 8 — CLASSIFY VERIFICATION: AGENT-OFFLINE vs USER-ONLY.
Before the prompt, state which checks are USER-ONLY:
  □ Benchmark conditions: machine idle, no other GPU or CPU workload, not thermally throttling,
    not on battery. Required before any timed run is valid.
  □ Profiler access: whether Nsight Compute counter collection is permitted (may need the NVIDIA
    control panel setting or elevation on Windows).
  □ Any run on a device other than the primary.
  □ Any toolchain install needed before the session.
Everything else — authoring, compiling, unit tests, the correctness gate — is AGENT-OFFLINE and
the agent runs it itself.

RULE 9 — REAL ARTIFACTS BEAT DOCUMENTS.
Model architecture values come from the config file shipped with the weights, not TECHNICAL_SPEC
§1. Hardware comes from runtime queries and Stage 0 microbenchmarks, not from any written figure.
Stage scope comes from TECHNICAL_SPEC §3. Where sources disagree, flag and defer to the artifact.

RULE 10 — AGENT SELF-RUNS THE OFFLINE GATE; HARD CHECKPOINT; COMMITS ONLY ON MY GO.
The agent writes the prediction into MEASUREMENTS.md and commits it FIRST. Then authors all files
and runs every AGENT-OFFLINE check itself: unit tests, clean build, correctness gate against the
PyTorch reference.
Iron rule: run the tests covering every file you TOUCH, not only files you create.
Timer: 5-minute per-test cap, self-enforced. Exceeded means treat as a hang: stop, inspect, fix,
rerun. On failure likewise. Fix-driven reruns only, no blind looping. **Stage 2's naive
implementation is SLOW BY DESIGN** — a long runtime there is not a hang; use reduced token counts
for correctness tests and state that you did.
When the offline gate is green, the agent STOPS. No benchmarks, no commit, no push, no PR. It
hands me ONLY the RULE 8 user-only checks as a HARD CHECKPOINT and waits.
  - If a user-only check FAILS: fix, re-run the offline gate, re-present. Still no commit.
  - On my "proceed": THEN, in order — (1) run the benchmark suite under BENCHMARK_PROTOCOL.md
    conditions; (2) if std dev exceeds 5% of median, declare the run INVALID, report it, stop —
    do not average it away or silently retry; (3) for GPU stages, collect the Nsight Compute
    counters listed in BENCHMARK_PROTOCOL.md §6; (4) fill the Measurement section of this stage's
    MEASUREMENTS.md entry from actual results, never estimated; (5) draft the Gap section with
    predicted vs measured, the mechanism, and **the specific counter that evidences it** —
    where the counters do not distinguish between candidate causes, write that plainly rather
    than picking one; (6) append future-relevant flags to PERSISTENT.md in the existing entry
    shape and bump "Last updated:"; (7) append this stage's concepts to LEARNING.md under its
    stage heading, marked `unread`, including any concept encountered that is not already listed;
    (8) git add -A; (9) git ls-files > repo-files.txt; (10) git add repo-files.txt; (11) commit
    descriptively; (12) push; (13) open a PR via the GitHub MCP tool. Report the PR URL.

RULE 11 — CC PROMPT BODY MUST MATCH THE CANONICAL FORMAT.
Per CC_PROMPT_FORMAT.md. Header `Task: Stage [N] — [title] ([one-line scope]).` then 2–3
sentences of scope naming what NOT to do. Delimiters `== CAPS ==`. Fixed order: PREDICTION
(COMMIT FIRST) → AUTHORITATIVE CONTRACT → [STAGE-SPECIFIC GAP/HEADLINE] → DETERMINE FROM THE LIVE
REPO → OUTPUTS → CONSTRAINTS RECAP → AFTER YOU BUILD. Plain declarative build-voice. Do NOT carry
my chat behavioral rules into the prompt: no source tags, no terse mode, no objection-first, no
"I don't know is valid".

RULE 12 — RECON-THEN-BUILD IS THE DEFAULT; PLAN MODE IS AN EXPLICIT OPT-IN.
Insert a plan-mode gate only when I explicitly say "plan mode" for that stage. **Exception:
Stage 10 defaults to plan mode** — the performance model's structure is a design decision worth
seeing before it is built, and RULE 14(e) depends on it being right.

RULE 13 — OPEN DECISIONS RESOLVE AS BUILD-TIME DETERMINATIONS, NOT APPROVAL GATES.
With plan mode OFF: "DETERMINE the convention from the live repo, pick accordingly, build it, and
STATE which you chose and why." if-X-then-Y-else-Z form. No STOP-for-approval unless opted in.

RULE 14 — PREDICTION BRIEFING.
Before the prompt, produce a briefing with these parts:
  (a) WHAT YOU ARE PREDICTING — the specific quantity, prefill and decode separated.
  (b) THE MINIMUM YOU NEED TO KNOW — the smallest set of concepts needed for a defensible guess.
      Fifteen to thirty minutes of reading, not a syllabus. Explain plainly, assuming no prior
      knowledge. I am not learning this material during the project; I need exactly enough to
      guess, and no more.
  (c) THE REASONING SHAPE — the form a good answer takes, with the relevant HARDWARE.md figures
      named, WITHOUT giving the answer.
  (d) THE RANGE CHECK — after I give my guess, say whether it is defensible and why, what a
      wildly wrong guess would imply, and what result would falsify my reasoning.
  (e) FROM STAGE 11 ONWARD — the prediction comes from running the Stage 10 performance model,
      not from me. The briefing instead states which model terms apply and what the model outputs.
      This is the model's prospective test and the reason it exists.
Neither of us knows the true answer; nothing has been run. Helping me reason toward a guess is
legitimate. Supplying the guess is not. If I ask you to just write it, refuse and walk me through
(b) instead.

RULE 15 — PREDICTION IS NOT THE AGENT'S JOB.
Claude Code never writes, revises, improves, or suggests a prediction. It transcribes mine
verbatim and commits it before building. If a performance stage's prompt arrives without a
prediction block, Claude Code STOPS and reports. A prediction produced by the thing being
measured is worthless.

RULE 16 — NO UNEARNED NUMBERS.
Claude Code never estimates, projects, extrapolates, or rounds a performance figure. Every number
in MEASUREMENTS.md and every number in a final report comes from a run that happened. If a run
did not happen, the field stays empty and the report says so. Self-relative speedups against the
project's own naive baseline are intermediate detail, never a headline figure.

RULE 17 — GAP EXPLANATIONS REQUIRE COUNTER EVIDENCE.
For any GPU stage, a gap explanation must name the Nsight counter supporting it and give that
counter's value. Where the counters do not distinguish between candidate causes, Claude Code
writes "the counters do not establish which of X or Y dominates" rather than selecting one. An
honest unestablished cause is correct output; a plausible invented mechanism is the failure this
project exists to avoid.

RULE 18 — NO sm_80+ INTRINSICS.
The primary device is Turing sm_75. `cp.async`, `ldmatrix`, and `mma` are unavailable. Stage 9
flash attention uses plain shared-memory staging. Any prompt or plan proposing these is wrong and
must be corrected before delivery. If a secondary-device experiment needs them, it is scoped and
labelled separately and never enters the main waterfall.

RULE 19 — BACKGROUND LOAD IS A RECORDED RUN CONDITION, NOT A REMINDER.
Every prompt for a stage that produces a timed number instructs the agent to enumerate,
at the hard checkpoint, every process holding a GPU context and every significant CPU
consumer, naming each, and to present that as an explicit close-these list rather than a
generic "ensure the machine is idle" line. Overlay and monitoring software is named
directly: Dragon Center and its services, NVIDIA App and its overlay, Nahimic, Discord,
browsers and embedded webviews, Steam, Logitech Options+, OneDrive.
Some consumers cannot be closed and must instead be RECORDED: Windows Defender
(MsMpEng), Riot Vanguard (vgc, vgtray), WmiPrvSE, SearchIndexer, nvcontainer, dwm, and
Claude Code itself. Stage 0 measured a 4.4 percent p90 noise floor with these present,
and roughly half of all configurations were invalid in at least one of two runs. That
population is therefore part of the measurement condition, and it is written into the
stage's MEASUREMENTS.md entry per BENCHMARK_PROTOCOL.md section 4, where an unrecorded
deviation is a fabricated number.
The agent re-enumerates AFTER the operator says proceed and BEFORE the first timed run,
and reports any process still present. It never closes anything itself.
If the recorded set differs from Stage 0's, the agent states the difference explicitly,
because the noise floor was measured against Stage 0's set and a changed set means the
floor may no longer apply.

RULE 20 — GIT SUCCESS IS VERIFIED BY STATE, NOT BY EXIT CODE.
A push is confirmed with git ls-remote against the branch, not by a zero exit status.
Stage 0 observed a push that hung on an interactive credential prompt while the wrapper
reported exit 0 and the log read "fatal: could not read Username", caused by Git
Credential Manager resolving the unqualified remote host to a stored account that was
not the repository owner. The agent verifies the remote ref matches the local HEAD and
reports both hashes before declaring the push done.

RULE 21 — DEFERRED WORK ITEMS ARE OWNED, NOT REMEMBERED.
PERSISTENT.md §7 is cleared and rewritten each session; §8 is not. §8 is where an item owned by a
stage more than one session away lives, so that it survives the clearing. Every kickoff reads §8
and reports the owned, touchable and irrelevant items by ID before the prompt is written.

For an item THIS stage owns, the Claude Code prompt carries it as an explicit instruction naming
its ID, and the stage's final report states what happened to it. Three outcomes are legitimate:
closed, with the evidence; still open, with the Status line updated to say what this session
learned; or explicitly deferred, with the reason. An item that silently disappears is not an
outcome — it is the failure this section exists to prevent, and it is the reason a work item owned
by Stage 11 must not sit in a section that Stage 1 erases.

For an item this stage could TOUCH but does not own, the prompt says so and forbids the agent from
acting on it unasked. A stage that half-closes another stage's item without being asked destroys
the evidence that item was waiting for.

An item a stage attempts and fails to close is closed as an item only when the reason it cannot
close is better-evidenced than the reason it was opened. Leaving a field empty with a real
diagnosis behind it is a valid result; substituting a plausible figure to make an item go away is
not, per RULE 16.

Any new work item a stage raises that is owned by a stage more than one session away goes into §8
with a new ID, not into §7. §7 may carry a one-line pointer to the ID. Never duplicate the text in
both — a fact recorded twice will eventually disagree with itself.