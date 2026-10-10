#!/usr/bin/env bash
# Installs the Linux StarCraft II binary (4.10) and the SMAC / SMACv2 maps.
# Downloading StarCraft II means you accept the Blizzard AI and Machine Learning License:
# https://blzdistsc2-a.akamaihd.net/AI_AND_MACHINE_LEARNING_LICENSE.html
set -euo pipefail

SC2PATH="${SC2PATH:-$HOME/StarCraftII}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [ ! -d "$SC2PATH" ]; then
    echo "Installing StarCraft II 4.10 to $SC2PATH"
    wget -q --show-progress -O "$TMP/SC2.4.10.zip" http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
    unzip -q -P iagreetotheeula "$TMP/SC2.4.10.zip" -d "$TMP"
    mv "$TMP/StarCraftII" "$SC2PATH"
else
    echo "StarCraft II found at $SC2PATH"
fi

# The SMACv2 map pack contains the SMAC maps as well as the 10gen_* / 32x32_* maps.
echo "Installing SMAC maps to $SC2PATH/Maps/SMAC_Maps"
wget -q -O "$TMP/SMAC_Maps.zip" https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip
unzip -q -o "$TMP/SMAC_Maps.zip" -d "$TMP/maps"
mkdir -p "$SC2PATH/Maps/SMAC_Maps"
find "$TMP/maps" -name '*.SC2Map' ! -name '._*' -exec cp {} "$SC2PATH/Maps/SMAC_Maps/" \;

echo "Done. Add this to your shell profile:"
echo "  export SC2PATH=$SC2PATH"
