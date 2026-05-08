import pandas as pd
from src.model_training import Training
from src.data_preprocessing import Prepare_sentiment_data
import logging

logging.basicConfig(
    level = logging.DEBUG,
    format  = "%(asctime)s - %(levelname)s - %(message)s"
)


def train_and_evaluate():
    try:
        train_dataset = torch.load(Train_Data)
        test_dataset = torch.load(Test_Data)
        train = Training()
        trainer = train.model_training(train_dataset, test_dataset)
        results = train.model_evaluation(trainer)

        print(results)
        Pusher = modelPusher()
        Pusher.updated_model_pusher(trainer, results)
        return results, trainer
    except Exception as e:
        logging.error(f"error occurred while training and evaluating the model {e}")

train_and_evaluate()