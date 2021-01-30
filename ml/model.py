from utils.utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-split")
parser.add_argument("--model")
args = parser.parse_args()

def DataManager(d:DataHandler=None, fr: FeatureRecipe=None, fe:FeatureExtractor=None):
    """
        Fonction qui lie les 3 premières classes de la pipeline et qui return FeatureExtractor.split(0.1)
    """
    data = DataHandler()
    data.get_process_data()
    recipe = FeatureRecipe(data.data)
    recipe.prepare_data()
    Fextractor = FeatureExtractor(recipe.data)
    return Fextractor.split(float(args.split))
 
X_train, X_test, y_train, y_test = DataManager()
m = ModelBuilder() 
m.train(X_train, y_train, args.model)
m.print_accuracy(X_test, y_test)
m.predict_test(X_test)
m.save_model('/home/lilian/project_cloud_computing/ml/')