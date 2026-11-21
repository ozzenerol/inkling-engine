#!/usr/bin/env python3
"""Compare build/tok_main (C++ tokenizer) with HF tokenizers on a mixed corpus. Owner: C."""
import glob, json, random, subprocess, sys
from tokenizers import Tokenizer
MODEL = "/Users/ozzenerol/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit"
tk = Tokenizer.from_file(MODEL + "/tokenizer.json")
rng = random.Random(7)
texts = ["", " ", "hello world", "Hello World!!", "  leading", "trailing   ", "a\n\nb\r\n\r\nc", "\t\tx", "1234567890 3.14159",
         "don't I'LL we're", "ÀÉÎõü naïve café", "日本語のテキスト、中文字符。", "emoji 🎉👍🏽 family 👨‍👩‍👧", "é combining",
         "nbsp x y　z\u0085w​q﻿k", "<|message_user|><|content_text|>hi<|end_message|>",
         "<|unknown|> <|end_message", "x<|content_text|>y", "#include <stdio.h>\nint main(){return 0;}\n",
         "    def f(self, x):\n        return x ** 2  # comment\n", "URL https://example.com/a?b=c&d=e", "ABCdefGHI camelCaseWord",
         "\u0000\u0001 control", "     \n     \n", "ⅫⅣ ½ ² ٣٤٥ १२३", "Ελληνικά Русский עברית العربية", "a" * 5000, "😀" * 300,
         " " * 1000 + "x", "\n" * 50]
for p in sorted(glob.glob("/Users/ozzenerol/projects/inkling-engine/**/*.*", recursive=True))[:200]:
    if p.endswith((".py", ".cpp", ".h", ".md", ".sh", ".json", ".mm", ".metal", ".txt")):
        try: texts.append(open(p, encoding="utf-8").read()[:200000])
        except Exception: pass
for p in glob.glob("/Users/ozzenerol/projects/saetta-macqwen/src/**/*.rs", recursive=True)[:60]:
    texts.append(open(p, encoding="utf-8").read())
alphabet = [chr(c) for c in list(range(32, 127)) + [0xa0, 0xe9, 0x2028, 0x3000, 0x1f600, 0x4e2d, 0x301, 0x85, 0xd, 0xa, 9,
                                                     0x2160, 0x660, 0x5d0, 0x200d, 0x1f3fd, 0x1d400, 0xb2]]
for _ in range(3000):
    texts.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 60))))
try:
    meta = json.load(open("/Users/ozzenerol/projects/inkling-engine/ref_data/meta.json"))
    for v in (meta if isinstance(meta, list) else meta.get("prompts", [])):
        if isinstance(v, dict):
            for k in ("prompt", "text", "rendered"):
                if isinstance(v.get(k), str): texts.append(v[k])
except Exception as e:
    print("no ref meta:", e)
want = [tk.encode(t, add_special_tokens=False).ids for t in texts]
r = subprocess.run([sys.argv[1] if len(sys.argv) > 1 else "build-c/src/server/tok_main"], input=json.dumps(texts), capture_output=True, text=True)
print(r.stderr.strip())
got = json.loads(r.stdout)
bad = [i for i in range(len(texts)) if want[i] != got[i]]
for i in bad[:5]:
    print("MISMATCH", repr(texts[i][:200]))
    j = next(k for k in range(min(len(want[i]), len(got[i])) + 1) if k >= min(len(want[i]), len(got[i])) or want[i][k] != got[i][k])
    print("  at", j, "want", want[i][j:j+8], [tk.id_to_token(x) for x in want[i][j:j+8]])
    print("       got ", got[i][j:j+8], [tk.id_to_token(x) for x in got[i][j:j+8]])
print(f"{len(texts)} texts, {sum(map(len, want))} tokens: {len(texts) - len(bad)} match, {len(bad)} mismatch")
sys.exit(1 if bad else 0)
