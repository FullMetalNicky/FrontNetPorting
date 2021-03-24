mkdir -p PenguiNet/golden

#QTrain
CUDA_VISIBLE_DEVICES=2 python3 Qtraining.py --regime regime.json --epochs 10 --load-trainset "../Data/160x96OthersTrain.pickle" --load-model "../Models/PenguiNet160x96_32c.pt" --save-model "QModel.pt" --quantize

#QDeploy
CUDA_VISIBLE_DEVICES=2 python3 QDeploy.py --regime regime.json --epochs 10 --load-trainset "../Data/160x96OthersTrain.pickle" --load-model "../Models/QModel.pth" --quantize

#QTest
CUDA_VISIBLE_DEVICES=2 python3 QTesting.py --regime regime.json --epochs 10 --load-testset "../Data/160x96OthersTest.pickle" --load-model "../Models/QModel.pth" --quantize