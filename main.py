from data_helpers import DataHolder
from configuration import Configuration
from train import Trainer
from utils import set_seeds, save, load, plot_prediction, plot_loss, calculate_RMSE, calculate_MAPE
from lstm import LSTMModel

def main():
    print("="*80)
    
    set_seeds(42)
    config_main = Configuration(preprocessing_method='none') # pca # en # none
    holder = DataHolder(config_main)

    train_dataset, eval_dataset, test_dataset = holder.get(
        dataset_index = 0,
    )

    # Initialize Trainer with the configuration and datasets
    print("="*80)
    trainer = Trainer(
        config = config_main,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        test_dataset = test_dataset,
    )

    # Train model
    forecast_model, training_losses, validation_losses = trainer.train(
        model = config_main.model,
        optimizer = config_main.optimizer,
    )
    plot_loss(training_losses, validation_losses)
    # save(forecast_model, "LSTM.pkl")

    # Make predictions on the test dataset
    forecast_model = load("LSTM.pkl")
  
    y_pred, y_true = trainer.predict(forecast_model)
    plot_prediction(y_pred, y_true)
    calculate_RMSE(y_pred, y_true)
    calculate_MAPE(y_pred, y_true)
    
if __name__=="__main__":
    main()
    print("Done!")

