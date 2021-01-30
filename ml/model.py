from utils.utils import *

def DataManager(d:DataHandler=None, fr: FeatureRecipe=None, fe:FeatureExtractor=None):
    """
        Fonction qui lie les 3 premières classes de la pipeline et qui return FeatureExtractor.split(0.1)
    """
    data = DataHandler()
    data.get_process_data()
    recipe = FeatureRecipe(data.data)
    recipe.prepare_data()
    Fextractor = FeatureExtractor(recipe.data)
    return Fextractor.split(0.1)
 
X_train, X_test, y_train, y_test = DataManager()
m = ModelBuilder() 
m.train(X_train, y_train)
m.print_accuracy(X_test, y_test)
m.predict_test(X_test)
m.save_model('/home/lilian/project_cloud_computing/ml/')