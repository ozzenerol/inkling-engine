# A: prompt-lookup drafts. copy.ids = a chat prompt asking to reproduce a source file (copy-heavy output).
PY=~/llm/inkling-venv/bin/python
[ -f ref_data/copy.ids ] || $PY -I - <<'PYEOF'
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(Path.home() / "llm/mlx-models/Inkling-2-Mini-MLX-4-8bit"))
src = Path("src/core/json.cpp").read_text()[:3500]
for name, q in (("copy", "Print this file again exactly as it is, in one code block, with no other text:\n\n```cpp\n" + src + "\n```"),
                ("edit", "Here is a file:\n\n```cpp\n" + src + "\n```\n\nRename the function `escape` to `escape_json` everywhere and print the whole file again in one code block.")):
    text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, reasoning_effort="none")
    np.array(tok.encode(text, add_special_tokens=False), np.int32).tofile(f"ref_data/{name}.ids")
PYEOF
for ids in copy edit p2; do
  for v in "plain" "dspark INK_NO_LOOKUP=1" "dspark"; do
    set -- $v; mode=$1; shift
    echo "-- $ids greedy: $v"
    env INK_SPEC_STATS=1 "$@" ./build/src/gpu/spec_check $mode ref_data/$ids.ids 400 4 $OUT/a-$ids-$mode${1:+-nolookup}.i32 2>&1 | grep -E "tok/s|dspark:|lookup:|adaptive"
  done
  cmp -s $OUT/a-$ids-plain.i32 $OUT/a-$ids-dspark.i32 && echo "$ids: lookup run token-identical to plain" || echo "$ids: lookup run DIFFERS from plain (T > 4 verify is not bit-identical to decode; check it is a near-tie)"
  echo "-- $ids temp 0.7: dspark"; env INK_SPEC_STATS=1 ./build/src/gpu/spec_check dspark ref_data/$ids.ids 400 4 $OUT/a-$ids-t07.i32 0.7 2>&1 | grep -E "tok/s|dspark:|lookup:"
done
