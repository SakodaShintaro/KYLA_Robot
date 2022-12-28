#!/bin/bash

set -eux

SCRIPT_DIR=$(dirname "$0")
cd ${SCRIPT_DIR}

rm -f FACE_FEATURES.db
python3 register_face_to_db.py hiraike ../assets/sample_images/kyla_members/hiraike0.jpg
python3 verify_db_data.py
