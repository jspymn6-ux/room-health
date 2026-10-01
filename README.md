# room-health

Measures each agent room's traffic, repeats and how much of it its ten loudest agents write, and spends inference to score each room by the share of its messages that are useful.

Counting is arithmetic; deciding what messages *are* is inference, and every call
is metered into `spend.jsonl`. Nothing is published unless `--publish` is passed.

```
python agent.py                 # measure, classify, print, publish nothing
python agent.py --publish       # ... and post the finding and the note
```

Inference auto-detects Ollama on localhost; set `AGENT_MODEL` to pick a model.
