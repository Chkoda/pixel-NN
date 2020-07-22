FOR /L %%i IN (4,1,10) DO python scripts/apply.py --input data/test.h5 --type number --output output --model modelWeights/kfold/theano104_kfold%%i.h5 --name theano_kfold%%i
