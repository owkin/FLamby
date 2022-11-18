import logging
import time

from fedml.core import ServerAggregator
from fedml.core.alg_frame.client_trainer import ClientTrainer  # isort
from torch.optim import SGD

from flamby.datasets.fed_heart_disease import LR, BaselineLoss, metric
from flamby.utils import evaluate_model_on_tests


class HeartDiseaseTrainer(ClientTrainer):
    def __init__(self, model, args=None):
        super().__init__(model, args)
        # We have to count in epochs
        self.epochs = 10

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        logging.info("Start training on Trainer {}".format(self.id))
        epochs = args.epochs  # number of epochs
        loss_func = BaselineLoss()
        optimizer = SGD(self.model.parameters(), lr=LR)
        since = time.time()
        # To draw loss and accuracy plots
        training_loss_list = []
        training_accuracy_list = []

        logging.info(" Train Data Size " + str(len(train_data.dataset)))
        # logging.info(" Test Data Size " + str(dataset_sizes["test"]))
        for epoch in range(epochs):
            logging.info("Epoch {}/{}".format(epoch, epochs - 1))
            logging.info("-" * 10)

            running_loss = 0.0
            accuracy = 0.0
            self.model.train()  # Set model to training mode

            # Iterate over data.
            for idx, (X, y) in enumerate(train_data):
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                y_pred = self.model(X)
                loss = loss_func(y_pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                accuracy += metric(
                    y_pred.detach().cpu().numpy(), y.detach().cpu().numpy()
                )

            epoch_loss = running_loss / len(train_data.dataset)
            epoch_accuracy = accuracy / len(train_data.dataset)

            logging.info(
                "Training Loss: {:.4f} Training Accuracy: {:.4f} ".format(
                    epoch_loss, epoch_accuracy
                )
            )
            training_loss_list.append(epoch_loss)
            training_accuracy_list.append(epoch_accuracy)

        time_elapsed = time.time() - since
        logging.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        logging.info("----- Training Loss ---------")
        logging.info(training_loss_list)
        logging.info("------Training Accuracy ------")
        logging.info(training_accuracy_list)
        return self.model


class HeartDiseaseAggregator(ServerAggregator):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        pass

    def _test(self, test_data, device):
        logging.info("Evaluating on Trainer ID: {}".format(self.id))

        test_metrics = {"test_correct": 0, "test_total": 0, "test_loss": 0}

        if not test_data:
            logging.info("No test data for this trainer")
            return test_metrics

        test_metrics = evaluate_model_on_tests(self.model, [test_data], metric)

        logging.info(f"Test metrics: {test_metrics}")
        return test_metrics

    def test_all(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        return True
