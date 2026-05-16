# This directory stores extracted MediaPipe landmark CSV files.
# Raw video files are NOT committed (see .gitignore).
#
# Structure:
#   data/landmarks/{word}/{index:04d}.csv
#   Each CSV: 30 rows (frames) x 126 columns
#            = [LEFT 21 landmarks × 3 axes | RIGHT 21 landmarks × 3 axes]
#   미감지 손은 zero-pad. 양손 모두 미감지된 프레임은 시퀀스에 포함되지 않는다.
