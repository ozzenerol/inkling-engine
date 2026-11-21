#!/bin/bash
# Status of the three headless agents: alive, event count, last tool/text.
cd ~/projects/inkling-engine || exit 1
echo "running agents: $(pgrep -f 'claude -p' | wc -l | tr -d ' ')"
for r in A B C; do
  f=logs/agent-$r.jsonl
  n=$(wc -l < "$f" | tr -d ' ')
  last=$(tail -40 "$f" | ~/llm/inkling-venv/bin/python -c '
import sys, json
out = []
for line in sys.stdin:
    try: d = json.loads(line)
    except Exception: continue
    if d.get("type") == "assistant":
        for c in d["message"].get("content", []):
            if c.get("type") == "tool_use":
                i = c.get("input", {})
                out.append("tool " + c["name"] + ": " + str(i.get("command") or i.get("file_path") or i.get("to") or "")[:110])
            elif c.get("type") == "text" and c["text"].strip():
                out.append("says: " + c["text"].strip().replace("\n", " ")[:160])
    elif d.get("type") == "result":
        out.append("FINISHED: " + str(d.get("subtype")) + " " + str(d.get("result", ""))[:200])
print("\n     ".join(out[-3:]) if out else "(starting)")
' 2>/dev/null)
  err=$(tail -c 200 logs/agent-$r.err)
  echo "[$r] $n events | $last"
  [ -n "$err" ] && echo "     err: $err"
done
echo "repo: $(git log --oneline 2>/dev/null | wc -l | tr -d ' ') commits, $(find . -path ./logs -prune -o -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.mm' -o -name '*.metal' \) -print | wc -l | tr -d ' ') source files"
[ -f NOTES.md ] && { echo "NOTES.md (last lines):"; tail -8 NOTES.md; }
