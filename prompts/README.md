# prompts

Drop markdown files in here and they become reusable prompts in the harness.

- `/prompts` — list available prompts
- `/prompt <name> [extra text...]` — send a prompt as a chat message
  (auto-starts a chat if none is active). `/p` is a short alias.

Files are matched by filename stem (without the `.md`). The full file
contents are sent as the user message; anything you type after the name
is appended on a new paragraph, so you can parameterize a prompt at the
call site without editing the file:

```
/p explain-code

(paste code here)
```

This README is filtered out of the listing only by convention — name your
prompts something meaningful.
