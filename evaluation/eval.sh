# art-fid
# cd projects/StyTR-2/evaluation;

CUDA_VISIBLE_DEVICES=0
python eval_artfid.py --sty  stylePath  --cnt contentPath --tar stylizedPath
