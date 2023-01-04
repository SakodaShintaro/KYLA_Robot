#!/bin/bash

set -eux

SCRIPT_DIR=$(dirname "$0")
cd ${SCRIPT_DIR}

rm ../assets/database/FACE_FEATURES.db 
python3 register_face_to_db.py
python3 verify_db_data.py
