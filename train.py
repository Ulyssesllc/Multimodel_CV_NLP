from model import ImageEncoder, TextEncoder, FusionModule
from process_data import MyData 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
# def __init__(self, csv_file, img_file):
csv_file = "training_data.csv"
img_dir = "Multimodals"
dataset = MyData(csv_file, img_dir)

train_dataset, test_dataset = train_test_split(dataset, train_size = 0.8)