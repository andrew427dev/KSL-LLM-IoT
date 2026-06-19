# This directory stores extracted MediaPipe landmark CSV files.
# Raw video files are NOT committed (see .gitignore).
#
# Structure:
#   data/landmarks/{word}/{index:04d}.csv       (직접 촬영: collect_data.py)
#   data/landmarks/{word}/aihub_{index:04d}.csv (AI Hub 변환: convert_aihub.py)
#   Each CSV: 30 rows (frames) x 131 columns
#            = [LEFT 21 landmarks × 3 axes | RIGHT 21 landmarks × 3 axes
#               | wrist-to-wrist vector 3 | presence flag 2]
#   각 손은 손목 상대좌표를 intra-hand scale(‖landmark9−landmark0‖)로 나눠
#   정규화한다. 미감지 손은 zero + presence flag 0. 양손 모두 미감지된
#   프레임은 시퀀스에 포함되지 않는다. 레이아웃 정의: src/feature_format.py
