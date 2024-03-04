import os

abs_path = os.path.abspath(os.path.dirname(__file__))
abs_path = os.path.dirname(abs_path)

def configs(**kwargs):
    name = kwargs.get('name')
    UnfoldingMethods = kwargs.get('UnfoldingMethods', ['LISTA', 'CPSS', 'AMI'])
    DataMethods = kwargs.get('DataMethods', ['DCNN'])
    ModelMethods = kwargs.get('ModelMethods', ['MUSIC', 'MVDR', 'SBL', 'ISTA'])
    return name, UnfoldingMethods, DataMethods, ModelMethods


name, UnfoldingMethods, DataMethods, ModelMethods = configs()
