from DatasetCreator import DatasetCreator


dc = DatasetCreator('../data/omarhand.bag')
#dc.CreateBebopDataset(0, True, "trainHand.pickle")
dc.CreateHimaxDataset(-1817123289, True, "trainHimaxHand.pickle")

