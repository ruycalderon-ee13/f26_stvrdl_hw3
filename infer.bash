python -u ./entry.py \
  --mode=infer \
  --data_path=../hw3-data-release \
  --checkpoint=../checkpoints/e_9_ap50_0.67.pt \
  --crop_size=512 \
  --score_threshold=0.05 \
  --mask_threshold=0.5 \
  --output_path=test-results.json