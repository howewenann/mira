---
name: skill-creator
description: Create, update, or validate effective MIRA skills with focused instructions and reusable resources. Use when the user asks to create, make, build, initialize, scaffold, improve, or validate a skill, or asks how MIRA skills should be designed.
license: MIT
compatibility: designed for MIRA
---

# Skill Creator

Create project skills that give MIRA reusable domain knowledge, workflows, and tool guidance.

## Start New Skills From This Template

For every new skill, begin with this concrete `SKILL.md` shape and fill it in:

```markdown
---
name: my-skill
description: Clear, specific description of what this skill does and when it should be used.
---

# Skill Name

## Overview

Brief explanation of the skill's purpose.

## When to Use

Conditions or requests where this skill applies.

## Instructions

1. Give concrete steps for the agent to follow.
2. Include important decision rules and constraints.
3. Explain how to complete the task correctly.

## Completion Criteria

Before finishing, verify that:

- [ ] The requested outcome has been produced.
- [ ] Required constraints have been followed.
- [ ] Any task-specific verification has passed.
```

YAML frontmatter alone is incomplete. Every skill requires a non-empty Markdown instruction body after the closing `---` delimiter.

The headings above are a recommended creation template, not validator requirements. A short legitimate body is valid without these exact headings. When updating an existing skill, read it first and improve it in place; do not force an otherwise valid skill into this heading structure.

## How to Fill the Template

### Choose the location and name

Create project skills at:

```text
.mira/skills/<skill-name>/SKILL.md
```

MIRA's packaged default skills are read-only. To customize a packaged default, create a project skill with the same frontmatter `name`; the project skill overrides the default.

Make `name` match the skill's parent directory exactly. Use at most 64 characters made from lowercase letters, digits, and single hyphens, with no leading, trailing, or consecutive hyphens.

### Write discovery metadata

Make `description` clearly explain both:

- what the skill does;
- when it should be used or triggered.

The description is always available during skill discovery, while the body is loaded only after the skill is selected. Put important trigger conditions in `description`, keep it at 1024 characters or fewer, and do not use angle brackets.

Use only these frontmatter fields: `name`, `description`, `license`, `compatibility`, `allowed-tools`, and `metadata`. When supplied, `compatibility` must be a string of at most 500 characters.

### Write concrete instructions

Replace the template placeholders with direct, imperative instructions. Include only what helps the agent perform the work:

- the skill's objective and intended result;
- ordered steps when sequence matters;
- decision rules for meaningful branches;
- important inputs, outputs, constraints, and pitfalls;
- verification and completion criteria.

Keep the instructions concise. Prefer a short example over general background, and match specificity to risk: flexible prose when several approaches are valid, ordered steps or pseudocode when a preferred sequence matters, and exact commands only when execution must be deterministic.

### Define completion criteria

`Instructions` describe how to do the work. `Completion Criteria` describe what must be true before finishing. Keep the criteria short, observable, checkable, and specific to the skill. Focus on the small set of important final-state checks that help the executing model decide whether the work is actually done.

Prefer checkable outcomes over vague quality statements. For example, avoid:

```text
- [ ] The result is high quality.
- [ ] The answer looks professional.
```

Prefer observable criteria such as:

```text
- [ ] All user-supplied constraints are reflected in the output.
- [ ] Required verification has completed successfully.
- [ ] Any unresolved blocker is explicitly identified.
```

Generate criteria appropriate to the specific skill rather than copying these examples into every skill. Use conditional wording when a criterion does not apply to every invocation. Do not duplicate every instruction as a criterion, and do not introduce requirements unsupported by the skill's purpose.

`Completion Criteria` is recommended authoring guidance only. It is not MIRA Goal Success Criteria, RubricMiddleware, an evaluator or judge, or a heading enforced by `validate_skill`. Preserve an existing skill's structure instead of adding this heading unless it is appropriate to the user's requested improvement.

### Add optional resources only when useful

Start with only `SKILL.md`. Do not create placeholder directories, unnecessary documentation files, or unused resources.

Add optional content only when it reduces repeated work or keeps the main instructions focused:

```text
skill-name/
├── SKILL.md
├── scripts/       # deterministic or repeatedly reused executable logic
├── references/    # detailed material loaded only when needed
└── assets/        # templates or concrete files used in generated output
```

Use progressive disclosure:

1. `name` and `description` support discovery.
2. The `SKILL.md` body provides the core workflow after selection.
3. Optional resources provide deeper detail only when the body says exactly when to load or use them.

### Clarify only material uncertainty

Before writing, identify representative requests, expected inputs and outputs, tools, constraints, and completion conditions. Use MIRA's `ask_user` tool only when material behavior, inputs, outputs, trigger conditions, or constraints are unclear. Ask the minimum focused questions needed; do not ask when the user's intent is already clear.

## Update Existing Skills In Place

Read the existing project skill and any resources its body references. Preserve useful structure and content while fixing ambiguity, missing constraints, repeated work, or unnecessary context. Do not modify packaged defaults and do not add resources merely to match the new-skill template.

## Validate Every Change

Validation is mandatory after creating or modifying a skill. Call:

```text
validate_skill(path=".mira/skills/<skill-name>")
```

If validation returns errors, address every returned repair action before calling `validate_skill` again. Do not rely on visual inspection of the YAML and do not finish while any validation error remains.

Finish only after validation returns `valid: true`.

Required loop:

```text
create or update SKILL.md
→ validate_skill
→ fix every returned error
→ validate_skill again
→ finish only when valid: true
```

After validation passes, check the skill against representative requests: its description should trigger for the intended use, its body should be actionable without hidden assumptions, optional resources should be discoverable, and every instruction should earn its context cost.
