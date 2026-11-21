#!/bin/zsh
# DSpark on/off identity + speed (owner B). Usage: tools/spec_check.sh IDS [N=200] [BLOCK=4]
set -e
cd "$(dirname "$0")/.."
ids=$1; n=${2:-200}; block=${3:-4}
tmp=$(mktemp -d)
INK_SPEC_STATS=1 ./build/src/gpu/spec_check plain "$ids" $n $block $tmp/plain.i32
INK_SPEC_STATS=1 ./build/src/gpu/spec_check dspark "$ids" $n $block $tmp/dspark.i32
if cmp -s $tmp/plain.i32 $tmp/dspark.i32; then echo "token streams IDENTICAL"; else echo "token streams DIFFER"; cmp $tmp/plain.i32 $tmp/dspark.i32 || true; fi
rm -rf $tmp
