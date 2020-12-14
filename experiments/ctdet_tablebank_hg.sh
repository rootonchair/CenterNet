cd src
# train

# first epoch
python main.py ctdet --dataset tablebank --data_dir /mnt/ai_filestore/home/vincent/TableBank/Detection --exp_id tablebank_hg --arch hourglass --batch_size 10 --master_batch 4 --lr 2.5e-4 --load_model ../models/ctdet_coco_hg.pth --gpus 1 --num_epochs 1 --lr_step 5,10,15

# second epoch
#python main.py ctdet --dataset tablebank --data_dir /mnt/ai_filestore/home/vincent/Detection --exp_id tablebank_hg2 --arch hourglass --batch_size 10 --master_batch 4 --lr 2.5e-8 --load_model ../models/model_last.pth --gpus 1 --num_epochs 1 --lr_step 5,10,15

# test
#python test.py ctdet --dataset tablebank --data_dir /mnt/ai_filestore/home/vincent/TableBank/Detection --exp_id tablebank_test --arch hourglass --load_model ../exp/ctdet/tablebank_hg/model_last.pth --trainval #--resume #--keep_res 

cd ..
