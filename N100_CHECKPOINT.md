checkpoints/manual_rotation_n100_d10_base10_moderight_brand1_trand1_it50000_bs128_lr1_seed0_20260320-015004.pt

python3 -u matrix_network_manual_rotation.py --n 100 --load-path checkpoints/manual_rotation_n100_d10_base10_moderight_brand1_trand1_it200000_bs64_lr0p5_seed0_20260318-220210.pt --iters 50000 --learning-rate 1.0 --eval-every 1000 --log-every 200 --device cpu --batch-size 128
