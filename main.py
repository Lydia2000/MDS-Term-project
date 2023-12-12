from data_helpers import DataHolder
from configuration import Configuration
from train import Trainer
from utils import save, load, plot_prediction, plot_loss
from lstm import LSTMModel

def main():
    print("="*80)

    config_main = Configuration()
    holder = DataHolder()
    train_dataset, eval_dataset, test_dataset = holder.get()
    
    # Initialize Trainer with the configuration and datasets
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
    save(forecast_model, "LSTM.pkl")

    # Make predictions on the test dataset
    forecast_model = load("LSTM.pkl")
  
    y_pred, y_true = trainer.predict(forecast_model)
    plot_prediction(y_pred, y_true)

if __name__=="__main__":
    main()
    print("Done!")

