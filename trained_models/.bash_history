dir
cd test/
dir
cd tor
dir
cd adapted_deep_embeddings/
dir
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_1_5/proto_net/opts.txt
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1=cuda10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_1_5/proto_net/opts.txt
logout
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1-cuda10.1 
module load tensorflow/1.14gpu
cd /home/z5141541/tor/adapted_deep_embeddings/
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_1_5/proto_net/opts.txt
dir
cd ..
dir
 cd few-shot/
dir
python train.py cd ..
cd ..
dir
cd few-shot/
dir
python3 ./train.py --dataset tinyImagenet --model protonet --method baseline
cd /srv/scratch/z5141541
dir
cd data/
dir
cd tinyImagenet/
dir
python3 write_tinyImagenet_filelist.py 
dir
mv train.json base.json
cd /home/z5141541/tor/
dir
cd few-shot/
dir
python3 write_tinyImagenet_filelist.py 
python3 ./train.py --dataset tinyImagenet --model protonet --method baseline
python3 ./train.py --dataset tinyImagenet --model Conv4 --method protonet
logout
cd tor/
dir
cd adapted_deep_embeddings/
module load python/3.7.3
module load cuda/10.1
module avil
module load cudnn/7.6.1-cuda10.1
module load pytorch/1.1.0
module load tensorflow/1.14gpu
dir
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_1_5/proto_net/opts.txt
qsub -I -l select=1:ncpus=4:ngpus=2:mem=46gb,walltime=2:00:00
cd ..
dir
qsub run.pbs 
qstat -u z5141541
qstat -f 181616
cat katana2:/home/z5141541/tor/run.pbs.o181616
cat /home/z5141541/tor/run.pbs.o181616
dir
qstat -f 181616
cat /home/z5141541/tor/run.pbs.o181616
dir
qstat -f 181616
ls
qstat -f 181616
ls
top
htop
man qstat
qstat -f 181616
ls
qstat -f 181616
dir
cd tor
qsub -I -l select=2:ncpus=4:ngpus=2:mem=46gb,walltime=2:00:00
qsub -I -l select=2:ncpus=4:ngpus=2:mem=46gb,walltime=8:00:00
qsub -I -l select=2:ncpus=4:ngpus=1:mem=46gb,walltime=8:00:00
python3 ./train.py --dataset tinyImagenet --model Conv4 --method protonet
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6-cuda10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
module load pytorch/1.1.0
cd tor/
dir
cd few-shot/
python3 ./train.py --dataset tinyImagenet --model Conv4 --method protonet
dir
cd tor
dir
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
dir
cd adapted_deep_embeddings/
dir
python3 run_model.py 
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_1_5/proto_net/opts.txt
cd ..
dir
logout
dir
cd tor
dir
rm run.pbs.e181616  run.pbs.o181616 
cd /home/z5141541/tor/
dir
cd adapted_deep_embeddings/
dir
cd trained_models/
dir
cd tiny_imagenet/
dir
cd tiny_imagenet_1_5
dir
cd proto_net/
dir
cd replication1/
dir
dir
qstat -f 181616
qstat -x 181616

cd tor
dir
cat run.pbs.o181616
cat run.pbs.e181616
qsub -I -l select=2:ncpus=4:ngpus=2:mem=46gb,walltime=8:00:00
head run.pbs 
head --line 15 run.pbs 
qsub run.pbs 
qstat -f 181731
qstat -f 181731
ls
cdto
cd tor/
ls
qstat -f 181731
htop
dir
cd adapted_deep_embeddings/
dir
cd ..
dir
ls
cd adapted_deep_embeddings/
qstat -f 181731
pbsnodes
man pbsnodes
pbsnodes a
pbsnodes -a
qstat
qstat -u z5141541
man qstat
qstat -u z5141541
qstat -f 181731
qsub -I -l select=2:ncpus=4:ngpus=2:mem=46gb,walltime=2:00:00
dir
cd tor
dir
cat run.pbs.e181731 
dir
cat run.pbs.o181731 
cd /srv/scratch/z5141541/data
dir
cd tinyImagenet/
cd ..
cd aptos/
dir
cd aptos/
dir
cd labels/
dir
head base15.json
ls
tail base15.json 
tail val15.json 
tail --line 50 val15.json 
head --line 15 val15.json 
cd /srv/scratch/z5141541
dir
cd dat
cd data
dir
cd aptos
dir
cd tor/
cd adapted_deep_embeddings/
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6-cuda10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
cd Des
dir
qsub -I -l select=2:ncpus=4:ngpus=2:mem=46gb,walltime=8:00:00
qsub -I -l select=1:ncpus=4:ngpus=2:mem=46gb,walltime=8:00:00
                                                                                                                                                                                                                                                                                                                                                                          dir
cd tor/
dir
qsub -I -l select=1:ncpus=4:ngpus=2:mem=46gb,walltime=8:00:00
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            10 base19.json 
dir
head trainLabels15.csv 
head trainLabels19.csv 
head testLabels19.csv 
head testLabels19.csv rm *.json
dir
rm *.json
dir
head --lines 20 trainLabels1
head --lines 20 trainLabels15.js
head --lines 20 trainLabels15.csv 
head --lines 20 trainLabels19.csv 
cd ../..
dir
python3 write_aptos2.py 
cd aptos/labels/
dir
head --lines 10 base15.json 
head --lines 10 base19.json 
rm *.json
cd ../..
dir
python3 write_aptos2.py 
cd aptos/labels/
dir
head --lines 10 base15.json 
cd /home/z5141541/tor/
dir
cd adapted_deep_embeddings/
dir
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_1_5/proto_net/opts.txt
cd /srv/scratch/z5141541/data/
dir
cd tinyImagenet/
dir
cd tiny-imagenet-200/
ls -l | wc -l
dir
cd train
ls -l | wc -l
dir
cd n01443537
ls -l | wc -l
dir
cd images
ls -l | wc -l
dir
cd tor
dir
cd adapted_deep_embeddings/
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
dir
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_1_5/proto_net/opts.txt
dmesg -T| grep -E -i -B100 'killed process'
logout
cd tor/
dir
cd adapted_deep_embeddings/
dir
qsub -I -l select=1:ncpus=8:ngpus=1:mem=46gb,walltime=8:00:00
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
module load python/3.7.3
module load cuda/7.6.1
module load cuda/10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
dir
cd tor/
cd adapted_deep_embeddings/
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_1_5/proto_net/opts.txt
cd tor
dir
cd adapted_deep_embeddings/
dir
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
cd Des
cd tor/
dir
cd adapted_deep_embeddings
dir
qsub -I -l select=2:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
man qsub
qstat -u z5141541
qdel 187910.kman.res
qdel 187910
qdel 187982
qstat -u z5141541
cd tor/
dir
cd adapted_deep_embeddings
dir
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
ir
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_1_5/proto_net/opts.txt
cd tor/adapted_deep_embeddings
cd ..
dir
cat run.pbs
mkdir records
dir
mv run.pbs.e181731 run.pbs.o181731 records/
cd records/
mkdir original
mv run.pbs.e181731 run.pbs.o181731 original/
cd ..
dir
qsub run.pbs 
qsub stat
qdel 190007
qsub stat
qdel 190007
qsub run.pbs 
qstat
qstat -u z5141541
dir
cd tor/
dir
cat run.pbs.o190008 
qstat
qstat -u z5141541
man qstat
qstat -f 190008
dir
cat run.pbs
cat run.pbs.e190008 
dir
rm run.[eo]*
rm run.pbs.[eo]*
dir
qsub run.pbs 
cd tor/
dir
qstat -u z5141541
dir
qstat
qstat -u z5141541
dir
cd tor
dir
qstat
qstat -u z5141541
htop
dir
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
qsub -I -l select=2:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
qsub -I -l select=2:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
qsub -I -l select=1:ncpus=8:ngpus=1:mem=46gb,walltime=8:00:00
qsub -I -l select=2:ncpus=8:ngpus=1:mem=46gb,walltime=8:00:00
dir
qsub run.pbs 
qstat
qstat -u z5141541
qdel 190224
qdel 191293
dir
qsub run.pbs 
qsub run2.pbs 
qsub run3.pbs 
qsub run4.pbs 
qsat
qstat
qstat -u z5141541
dir
cd tor
dir
cd tra
cd adapted_deep_embeddings
cd trained_models/
dir
cd tiny_imagenet/
dir
cd tiny_imagenet_1_5
dir
cd proto_net/
dir
cd replication1/
dir
ls -l
cat checkpoint 
ls
less model.ckpt-9201.meta 
less checkpoint 
pwd
cd ..
tensorboard
module list
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1-10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
tensorboard
tensorboard --log_dir replication1
tensorboard --logdir replication1
logout
qstat
qstat -u z5141541
cd tor/
dir
cd records/
dir
cd original/
cd ..
dir
cd ..
dir
cat run.pbs.o190224 
rm run.pbs.o190224 
rm run.pbs.e190224 
dir
qstat
qstat -u z5141541
cd Des
cd tor/
dir
cd adapted_deep_embeddings
dir
cd trained_models/
dir
cd tiny_imagenet/
dir
cd tiny_imagenet_10_10/
dir
cd proto_net/
tensorboard --logdir replication1
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
tensorboard --logdir replication1
qstat
qstat -u z5141541
cd tor
dir
cat run.pbs.o191305 
cat run.pbs2.o191305 
cat run2.pbs.o191306 
cd tor/
qsub -I -l select=1:ncpus=8:ngpus=1:mem=46gb,walltime=8:00:00
qstat
qstat -u z5141541
qdel 191307
qdel 191308
qsub -I -l select=1:ncpus=8:ngpus=1:mem=46gb,walltime=8:00:00
qstat
qstat -u z5141541
rm 192087
qdel 192087
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
cd tor/
dir
cd adapted_deep_embeddings
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
dir
cd tor
dir
logout
dir
cd tor/
dir
rm run*.pbs.*
dir
cd adapted_deep_embeddings
dir
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
logout
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
                                                                                                                                                                                                                                                                                                                                                                                                                                   qstat
qstat -u z5141541
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
                                                                                                                                                                                                                                                                                                                                cd tor/adapted_deep_embeddings
load python/3.7.3
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
dir
tensorboard --logdir="./graphs/" --port 6006
cd tor/
dir
cd adapted_deep_embeddings
dir
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
tensorboard --logdir="./graphs/" --port 6006
logout
cd tor/adapted_deep_embeddings
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
tensorboard --logdir="./graphs/" --port 6006
cd tor/
dir
cd adapted_deep_embeddings
dir
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
cd tor/adapted_deep_embeddings
dir
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
tensorboard --logdir="./graphs/" --port 6006
logout
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
tensorboard --logdir="./graphs/" --port 6006
cd tor/adapted_deep_embeddings
tensorboard --logdir="./graphs/" --port 6006
cd tor/
qstat
qstat -u z5141541
qdel 192928
dir
cd adapted_deep_embeddings
dir
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
                                                                                                                                                                                                                                                                                                                                                      module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
cd tor/adapted_deep_embeddings
tensorboard --logdir="./graphs/" --port 6006
logout
qstat -u z5141541
qdel 192993
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
cd tor/adapted_deep_embeddings
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
cd graphs/
rm *
cd ..
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_1_5/proto_net/opts.txt
qstat -u z5141541
dir
cd tor/
dir
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
qstat -u z5141541
dir
cd tor/adapted_deep_embeddings
dir
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
                                                                                                                                       tensorboard --logdir="./graphs/" --port 6006
module load python/3.7.3
module load cuda/10.1
module load cudnn/7.6.1-cuda10.1
module load tensorflow/1.14gpu
cd tor/adapted_deep_embeddings
tensorboard --logdir="./graphs/" --port 6006
qstat -u z5141541
head tor/refresh2.sh 
qstat -u z5141541
cd tor/
dir
source refresh2.sh 
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
dir
cd tor
dir
source refresh3.sh 
cd adapted_deep_embeddings
dir
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
cd tor/
source refresh2.sh 
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
qstat -u z5141541
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
cd ..
qstat -u z5141541
man qsub
qstat -u z5141541
logout
qstat -u z5141541
dir
cd tor/
source refresh3.sh 
cd adapted_deep_embeddings
tensorboard --logdir="./graphs/" --port 6006
cd /srv/scratch/z5141541/
dir
cd data/
dir
cd aptos/
dir
cat write_aptos.py
cd atos
dir
cd aptos/
dir
cd /home/z5141541
dir
cd tor/few-shot/
dir
cd data/
dir
cd ..
dir
cd filelists/
dir
cd aptos
dir
ls
ls -l
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
qstat -u z5141541
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
logout
dir
cd tor/
source refresh3.sh 
cd adapted_deep_embeddings
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
cd graphs/
rm *
cd ..
dir
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
cd graphs/
remove *
rm *
cd ..
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
cd graphs/
rm *
cd ..
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
cd graphs/
dir
rm *
cd ..
dir
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
cd graphs/
dir
rm *
cd ..
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
cd graphs/
rm *
cd ..
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
cd graphs/
dir
rm *
cd ..
dir
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
cd graphs/
rm *
cd ..
dir
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
cd graphs/
rm gra
rm *
cd ..
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
cd graphs/
rm *
cd ..
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
cd graphs/
dir
rm *
cd ..
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
cd graphs/
dir
rm *
cd ..
dir
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
cd graphs/
dir
rm *
cd ..
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
logout
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
dir
cd tor/adapted_deep_embeddings
dir
cd ..
source refresh3.sh 
cd adapted_deep_embeddings
dir
tensorboard --logdir="./graphs/" --port 6006
clear
tensorboard --logdir="./graphs/" --port 6006
cd trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/
dir
cat opts.txt 
cd ../../
cd ..
dir
cd ..
cd trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/
cd ../../../
ir
dir
cd ..
dir
tensorboard --logdir="./graphs/" --port 6006
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=8:00:00
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=4:00:00
dir
logout
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=2:00:00
qstat
qstat -u z5141541
qdel 195204
qstat -u z5141541
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=2:00:00
cd /srv/scratch/z5141541/data/
dir
cd aptos/
dir
cd ..
dir
cd tinyImagenet/
dir
cd tiny-imagenet-200/
dir
cd train
dir
cd tor/adapted_deep_embeddings
cd ..
source refresh3.sh 
cd adapted_deep_embeddings
dir
tensorboard --logdir="./graphs/" --port 6006
qstat -u z5141541
tensorboard --logdir="./graphs/" --port 6006
dir
cd models/
dir
cat proto_model.py 
cd tor/
source refresh3.sh 
cd adapted_deep_embeddings
dir
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
logout
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=2:00:00
cd dir
cd ..
dir
cd z5141541
dir
cd tor/
dir
qsub run.pbs 
qstat -u z5141541
dir
cat run.pbs.e195413 
qsub run.pbs 
rm run.pbs.*
qstat -u z5141541
cd tor/
source refresh3.sh 
dir
cd adapted_deep_embeddings
tensorboard --logdir="./graphs/" --port 6006
cd tor/
source refresh3.sh 
cd adapted_deep_embeddings
dir
tensorboard --logdir="./graphs/" --port 6006
cd tor/
dir
cd adapted_deep_embeddings
dir
cd ..
dir
cat run.pbs.o195415 
qsub run.pbs 
qstat -u z5141541
qdel 195455
qsub run.pbs 
qstat -u z5141541
qdel 195472
qstat -u z5141541
dir
cd tor/
dir
cat run.pbs.o195468 
dir
qsub run.pbs
qstat -u z5141541
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=2:00:00
cd tor/
source refresh3.sh 
tensorboard --logdir="./graphs/" --port 6007
cd adapted_deep_embeddings
tensorboard --logdir="./graphs/" --port 6007
cd tor/
dir
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=2:00:00
qstat -u z5141541
qdel 196059
dir
cd tor/
dir
cd adapted_deep_embeddings
dir
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=2:00:00
qstat -u z5141541
qdel 196055
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=2:00:00
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=4:00:00
qstat -u z5141541
dir
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=4:00:00
qstat -u z5141541
dir
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=4:00:00
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=2:00:00
dir
cd tor/
source refresh3.sh 
cd adapted_deep_embeddings
dir
tensorboard --logdir="./graphs/" --port 6007
tensorboard --logdir="./graphs/" --port 6006
cd
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=2:00:00
logout
cd tor/
source refresh3.sh 
cd adapted_deep_embeddings
dir
cd ..
cd adapted_deep_embeddings2/
python3 main.py @trained_models/tiny_imagenet/tin
dir
cd ..
cd adapted_deep_embeddings
clear
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
.
 
dir
python3 main.py @trained_models/tiny_imagenet/tiny_imagenet_10_10/proto_net/opts.txt
logout
dir
cd tor/
dir
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=4:00:00
qsub -I -l select=1:ncpus=16:ngpus=2:mem=92gb,walltime=4:00:00
cd tor/
source refresh3.sh 
dir
cd adapted_deep_embeddings
tensorboard --logdir="./graphs/" --port 6006
dir
cd tor/
dir
cat run.pbs 
qsub run.pbs 
logout
dir
qstat -u z5141541
dir
cd tor/
dir
cat run.pbs.o196355 
qsub run.pbs
cd tor/
source refresh3.sh 
dir
cd adapted_deep_embeddings
dir
tensorboard --logdir="./graphs/" --port 6006
logout
cd tor/
source refresh3.sh 
cd adapted_deep_embeddings
dir
tensorboard --logdir="./graphs/" --port 6006
cd tor/
source refresh3.sh 
cd adapted_deep_embeddings
dir
tensorboard --logdir="./graphs/" --port 6006
tensorboard --logdir="./graphs/" --port 6007
cd tor/
dir
cat run.pbs.o196508 
di
cat run.pbs.o196355 
qsub run.pbs
qdel 196571
cd grap
cd to
cd adapted_deep_embeddings
dir
mv graphs graphs_backup
mkdir graphs
cd ..
qsub run.pbs
qstat -u z5141541
dir
cd tor/
dir
cat run.pbs.o196572 
cd tor/
dir
cd gra
cd adapted_deep_embeddings
cd graphs
dir
cat run.pbs
cd ..
cat run.pbs
dir
cd ..
cat run.pbs
cd ..
dir
cd tp
dc tor/
dir
cd tor/
dir
rm run.pbs.o*
rm run.pbs.e*
dir
cd adapted_deep_embeddings/graphs
dir
rm events.out.tfevents.1581952440.k103 
rm events.out.tfevents.1581951021.k103 
dir
mkdir k10n5
mv events.out.tfevents.1581950046.k103 k10n5/
dir
cd ..
dir
qstat -u z5141541
cd ..
dir
cat run.pbs 
qsub run.pbs
cd Des
cd tor/
dir
source refresh3.sh 
cd adapted_deep_embeddings
tensorboard --logdir="./graphs/" --port 6007
tensorboard --logdir="./graphs/" --port 6006
cd tor/
dir
cd adapted_deep_embeddings
cd graphs
di
dir
qstat -u z5141541
cd ..
dir
cd ..
dir
qsub run.pbs 
qstat -u z5141541
vim run.pbs 
qsub run.pbs 
qstat -u z5141541
cd tor/adapted_deep_embeddings
cd graphs
dir
