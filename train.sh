export PD_WN25E="berteach_wn025"
export PD_WN5E="berteach_wn05"
export PD_WN75E="berteach_wn075"

# 0
export PD=${PD_WN25E}
python train.py

# 1
# export PD=${PD_WN5E}
# python train.py

# 2
# export PD=${PD_WN75E}
# python train.py

# 3
# export PD="test"
# python train.py