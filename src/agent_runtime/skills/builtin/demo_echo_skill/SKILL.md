---
name: demo_echo_skill
description: A demo skill used to verify built-in skill loading, discovery, and invocation. It accepts user text and returns a structured echo result.
triggers: [echo, demo, test]
---

# Demo Echo Skill

A minimal built-in skill for verifying the skill loading, discovery, and invocation pipeline.

## Usage

When the user sends text prefixed with "echo", return the text back as a structured echo result.

## Purpose

- Verify that system built-in skills load correctly from the `builtin/` directory
- Confirm that `list_skills` can discover this skill
- Test the skill invocation flow end-to-end without external dependencies
