#!/bin/bash

set -e -u

SCRIPT_DIR=$(cd $(dirname $0); pwd)

# cd into the build directory
cd "$SCRIPT_DIR"
cd ../build

# renew a temporary release directory
rm -rf tmp_release
mkdir -p tmp_release

# copy binary files into the temporary release directory
cp -R ./Wahwth_artefacts/Release/Standalone ./tmp_release
cp -R ./Wahwth_artefacts/Release/VST3 ./tmp_release
cp -R ./Wahwth_artefacts/Release/AU ./tmp_release

# prepare symlinks
ln -f -s "/Applications" "tmp_release/Standalone"
ln -f -s "/Library/Audio/Plug-Ins/VST3" "tmp_release/VST3"
ln -f -s "/Library/Audio/Plug-Ins/Components" "tmp_release/AU"

# copy resource files for release
cp "../resource/README-ja.pdf" "tmp_release"

# create zip archive
cd tmp_release
zip --symlink -r ../wahwth-release-mac.zip .
