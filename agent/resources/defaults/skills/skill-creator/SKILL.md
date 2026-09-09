---
name: skill-creator
description: Create, update, or validate effective MIRA skills with focused instructions and reusable resources. Use when the user asks to create, make, build, initialize, scaffold, improve, or validate a skill, or asks how MIRA skills should be designed.
license: MIT
compatibility: designed for MIRA
---

# Skill Creator

Create skills that give MIRA reusable domain knowledge, workflows, and tool guidance.

## Skill location and precedence

Create project skills at:

```text
.mira/skills/<skill-name>/SKILL.md
```

MIRA also ships packaged default skills. Packaged defaults are read-only. A project skill under `.mira/skills` overrides a packaged default with the same frontmatter `name`.

For a basic skill, create only the required `SKILL.md`. Do not create placeholder directories or files.

## Core principles

### Keep instructions concise

Assume the agent already understands general concepts. Include only specialized knowledge, repeatable procedure, important constraints, and non-obvious pitfalls. Prefer short examples over long explanations.

### Match specificity to risk

- Use flexible prose when multiple approaches are valid.
- Use ordered steps or pseudocode when a preferred sequence matters.
- Use exact commands or scripts only when execution is fragile or must be deterministic.

### Design for progressive disclosure

Skills have three possible levels:

1. Frontmatter `name` and `description`, always available for discovery.
2. The `SKILL.md` body, loaded when the skill applies or is invoked directly.
3. Optional bundled resources, loaded only when the instructions say they are needed.

Keep the body focused. When detailed reference material is genuinely necessary, put it in a clearly named optional resource and tell the agent exactly when to read it.

## Skill anatomy

The minimum valid structure is:

```text
skill-name/
└── SKILL.md
```

Advanced skills may add resources only when the task requires them:

```text
skill-name/
├── SKILL.md
├── scripts/       # deterministic or repeatedly reused executable logic
├── references/    # detailed material loaded on demand
└── assets/        # templates or files used in generated output
```

Do not scaffold `scripts/`, `references/`, or `assets/` by default.

## Required SKILL.md format

Start the file with YAML frontmatter:

```yaml
---
name: skill-name
description: Describe what the skill does and the requests or contexts that should trigger it.
---
```

Follow these authoring rules:

- Make `name` match the parent directory exactly.
- Use lowercase letters, digits, and single hyphens in `name`.
- Keep `name` at 64 characters or fewer.
- Make `description` explain both capability and trigger conditions.
- Keep `description` at 1024 characters or fewer and omit angle brackets.
- Use only `name`, `description`, `license`, `compatibility`, `allowed-tools`, and `metadata` in frontmatter.
- Keep `compatibility` at 500 characters or fewer when present.
- Write the body as direct imperative instructions.

The description is the primary automatic trigger. Put all important "when to use" language there because the body is not available until after discovery chooses the skill.

## Creation workflow

### 1. Understand the skill through examples

Identify concrete requests that should use the skill and requests that should not. Determine the expected inputs, outputs, tools, constraints, and completion conditions.

When the behavior, trigger conditions, inputs, outputs, or intended use are materially unclear, use MIRA's `ask_user` tool rather than guessing. Ask only the minimum focused questions necessary, and ask a small number at a time. Do not ask questions when the user's intent is already clear.

Useful clarification topics include:

- representative user requests;
- required output format;
- tools or external systems involved;
- constraints that must always hold;
- examples of failure or unwanted behavior.

### 2. Plan reusable content

For each example, identify the steps and knowledge the agent would otherwise have to rediscover. Decide whether concise instructions are sufficient.

Add a script only for logic that benefits from deterministic execution or would otherwise be rewritten repeatedly. Add a reference only for substantial detail that is needed in some cases but not every invocation. Add an asset only when the output must reuse a concrete file.

### 3. Create or update SKILL.md

Create `.mira/skills/<skill-name>/SKILL.md`, or edit the existing project skill. Do not attempt to modify a packaged default. To customize a default, create a project skill with the same name.

Organize the body around the actual workflow. Include:

- a concise objective;
- ordered steps where sequence matters;
- decision rules for meaningful branches;
- references to optional resources and when to load them;
- verification and completion criteria;
- important pitfalls.

Avoid generic motivational prose, duplicated background, and documentation files that are not needed to execute the skill.

### 4. Validate with the tool

After every creation or modification, call:

```text
validate_skill(path=".mira/skills/<skill-name>")
```

Do not rely on visually inspecting YAML. If validation returns errors, fix `SKILL.md` and call `validate_skill` again. Finish only after validation returns `valid: true`.

Required loop:

```text
create or update SKILL.md
→ validate_skill
→ fix every reported error if invalid
→ validate_skill again
→ finish only when valid
```

### 5. Review usefulness

Check the skill against the concrete examples:

- Does the description trigger on the intended requests?
- Does it avoid claiming unrelated requests?
- Can the agent follow the body without hidden assumptions?
- Are optional resources discoverable from the body?
- Are steps precise enough for the task's risk level?
- Is every section worth its context cost?

### 6. Improve from real use

After the skill is used, update it when actual behavior reveals ambiguity, missing constraints, repeated work, or unnecessary context. Re-run `validate_skill` after every update and finish only when it passes.
