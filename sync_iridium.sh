#!/bin/bash

rsync -av \
  --exclude='.git' \
  --exclude='.DS_Store' \
  --exclude='.vscode' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='cvvdp.egg-info' \
  --exclude='results' \
  ./ \
  iridium.cl.cam.ac.uk:/anfs/gfxdisp/xw506/Project/ColorVideoVDP/