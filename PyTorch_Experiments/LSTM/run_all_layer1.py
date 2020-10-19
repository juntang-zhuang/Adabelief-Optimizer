import os

# 1-layer lstm
cmd = ' python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 --optimizer adabelief --lr 0.001 --eps 1e-16 --eps_sqrt 0.0 --nlayer 1 --run 0'
os.system(cmd)


cmd = ' python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 --optimizer sgd --lr 30 --eps 1e-12 --nlayer 1 --run 0'
os.system(cmd)

cmd = ' python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 --optimizer adabound --lr 0.01 --eps 1e-8 --nlayer 1 --run 0'
os.system(cmd)

cmd = ' python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 --optimizer yogi --lr 0.01 --eps 1e-3 --nlayer 1 --run 0'
os.system(cmd)

cmd = ' python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 --optimizer adam --lr 0.001 --eps 1e-12 --nlayer 1 --run 0'
os.system(cmd)

cmd = ' python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 --optimizer msvag --lr 30 --eps 1e-8 --nlayer 1 --run 0'
os.system(cmd)

cmd = ' python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 --optimizer radam --lr 0.001 --eps 1e-12 --nlayer 1 --run 0'
os.system(cmd)

cmd = ' python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 200 --save PTB.pt --when 100 145 --clip 0.25 --beta1 0.9 --beta2 0.999 --optimizer adamw --lr 0.001 --eps 1e-12 --nlayer 1 --run 0'
os.system(cmd)



