import argparse  
from train import train, test   
from MLP import MLP_fusion
from Q_former import Q_former_fusion
from Contrastive import Q_cons_fusion
from attention_model import AttentionModule
MODEL_MAP = {
    "Q_former_fusion": Q_former_fusion,
    "MLP_fusion": MLP_fusion,
    "Q_cons_fusion":Q_cons_fusion,
    "AttentionModule" : AttentionModule
}
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type= str, default = "")
parser.add_argument("--model", type=str, default="Q_cons_fusion")
parser.add_argument("--epochs", type = int, default = 10)

if __name__ == "__main__":  # ← FIXED '=' to '=='
    args = parser.parse_args()

    # Get the actual model class
    model_class = MODEL_MAP[args.model]
    model_instance = model_class()  # Instantiate the model

    train(model=model_instance, 
          dataset=args.dataset,
          epoch=args.epochs)