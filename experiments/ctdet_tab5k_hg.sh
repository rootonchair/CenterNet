cd src
# train
#python main.py ctdet --dataset tab5k --exp_id coco_hg --arch hourglass --batch_size 2 --master_batch 4 --lr 2.5e-4 --load_model ../models/ctdet_coco_hg.pth --gpus 1 --num_epochs 20 --lr_step 5,10,15
# test
python test.py ctdet --dataset tab5k --exp_id tab5k_hg_test --arch hourglass --load_model ../exp/ctdet/coco_hg/model_best.pth --trainval #--resume #--keep_res 

# flip test
#python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume --flip_test 
# multi scale test
#python test.py ctdet --exp_id coco_hg --arch hourglass --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
