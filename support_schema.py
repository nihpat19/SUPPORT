import os
import datajoint as dj
dj.config['enable_python_native_blobs'] = True
if not "stores" in dj.config:
    dj.config['stores'] = {}
dj.config['database.host'] = os.environ['DJ_HOST']
dj.config['database.user'] = os.environ['DJ_USER']
dj.config['database.password'] = os.environ['DJ_PASS']
dj.config['stores']['minio'] = {
    'protocol': 'file',
    'location': '//jr-storage.ad.bcm.edu/jr-scratch03A/Nihil/trainedModels/'
}

model_schema_name = "nihil_nnfabrik_support_optuna"

dj.config['nnfabrik.schema_name'] = model_schema_name

from nnfabrik.main import my_nnfabrik
from nnfabrik.templates.trained_model import TrainedOptunaModelBase

nnfabrik_module = my_nnfabrik(
    model_schema_name,
    context=None,
    use_common_fabrikant=False
)
Fabrikant, Seed, Model, Dataset, Trainer = nnfabrik_module.Fabrikant, nnfabrik_module.Seed, nnfabrik_module.Model, \
nnfabrik_module.Dataset, nnfabrik_module.Trainer

schema = dj.Schema(model_schema_name)
@schema
class TrainedModel(TrainedOptunaModelBase):
    table_comment = "SUPPORT trained denoising model"
    nnfabrik = nnfabrik_module