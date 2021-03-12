from config import Config
from utils.train_test_split import TrainTestSplit

config = Config()
tts = TrainTestSplit(config)
training_volume, testing_volume = tts.split('images_volumes', 'item_seg', 'liver_seg')

