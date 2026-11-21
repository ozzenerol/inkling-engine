# A: prompt-lookup drafts on copy-heavy output. The prompt ends inside the answer's code block (no thinking), so
# the model is copying from the first generated token.
PY=~/llm/inkling-venv/bin/python
[ -f ref_data/copy.ids ] || $PY -I - <<'PYEOF'
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(Path.home() / "llm/mlx-models/Inkling-2-Mini-MLX-4-8bit"))
src = Path("src/core/json.cpp").read_text()
src = src[:src.rfind("\n}\n", 0, 5000) + 3]
for name, q in (("copy", "Print this file again exactly as it is, in one code block, with no other text:\n\n```cpp\n" + src + "```"),
                ("edit", "Here is a file:\n\n```cpp\n" + src + "```\n\nRename the function `utf8` to `put_utf8` everywhere and print the whole file again in one code block, with no other text.")):
    text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, reasoning_effort="none")
    text += "<|message_model|><|content_text|>```cpp\n"
    np.array(tok.encode(text, add_special_tokens=False), np.int32).tofile(f"ref_data/{name}.ids")
PYEOF
for ids in copy edit; do
  echo "-- $ids greedy plain";  ./build/src/gpu/spec_check plain ref_data/$ids.ids 600 4 $OUT/a-$ids-plain.i32 2>&1 | grep -E "tok/s"
  for v in "INK_NO_LOOKUP=1" "INK_LOOKUP_MAX=3" "INK_LOOKUP_MAX=7" "INK_LOOKUP_MAX=15"; do
    echo "-- $ids greedy dspark $v"
    env INK_SPEC_STATS=1 $v ./build/src/gpu/spec_check dspark ref_data/$ids.ids 600 4 $OUT/a-$ids-$v.i32 2>&1 | grep -E "tok/s|dspark:|lookup:|adaptive"
    cmp -s $OUT/a-$ids-plain.i32 $OUT/a-$ids-$v.i32 && echo "   token-identical to plain" || echo "   DIFFERS from plain"
  done
  echo "-- $ids temp 0.7 dspark (lookup max 3)"; env INK_SPEC_STATS=1 ./build/src/gpu/spec_check dspark ref_data/$ids.ids 600 4 $OUT/a-$ids-t07.i32 0.7 2>&1 | grep -E "tok/s|dspark:|lookup:"
done
