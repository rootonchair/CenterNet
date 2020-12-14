cd src
# train

# first epoch
python main.py ctdet --dataset tablebank --data_dir /mnt/ai_filestore/home/vincent/TableBank/Detection --exp_id tablebank_hg_test --arch hourglass --batch_size 10 --master_batch 4 --lr 2.5e-4 --load_model ../models/ctdet_coco_hg.pth --gpus 1 --num_epochs 1 --lr_step 5,10,15

# second epoch
#python main.py ctdet --dataset tablebank --exp_id tablebank_hg2 --arch hourglass --batch_size 10 --master_batch 4 --lr 2.5e-8 --load_model ../models/model_last.pth --gpus 1 --num_epochs 1 --lr_step 5,10,15

# test
#python test.py ctdet --dataset tablebank --exp_id tablebank_test --arch hourglass --load_model ../models/model_last.pth --trainval #--resume #--keep_res 
# flip test
#python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume --flip_test 
# multi scale test
#python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
