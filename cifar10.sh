nohup python utils/main.py --dataset seq-cifar10 --model robustdualvit --buffer_size 500 --lr 0.03 --minibatch_size 32 --alpha 0.3 --r_alpha 0.3 --mi_alpha 0.3 --batch_size 32 --n_epochs 1 > results/cifar10/robustdualvit_cifar10.out 2>&1 &

nohup python utils/main.py --dataset seq-cifar10 --model derppfense --buffer_size 500 --lr 0.03 --minibatch_size 32 --alpha 0.3 --beta 0.1 --batch_size 32 --n_epochs 1 > results/cifar10/derppadv_cifar10.out 2>&1 &

nohup python utils/main.py --dataset seq-cifar10 --model unifiedcladv --buffer_size 500 --lr 0.03 --minibatch_size 32 --alpha 0.3 --beta 0.1 --batch_size 32 --n_epochs 1 > results/cifar10/unifiedcladv_cifar10.out 2>&1 &

nohup python utils/main.py --dataset seq-cifar10 --model derpp --buffer_size 500 --lr 0.03 --minibatch_size 32 --alpha 0.3 --beta 0.1 --batch_size 32 --n_epochs 1 > results/cifar10/derpp_cifar10.out 2>&1 &
