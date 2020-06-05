#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

# cd into the build directory
cd "$SCRIPT_DIR"
cd ../resources

pandoc README-ja.md -o README-ja.pdf -V documentclass=ltjarticle --pdf-engine=lualatex --toc --variable urlcolor=cyan
