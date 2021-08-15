


from pytorch_lightning.utilities.cli import LightningCLI
from tkitTagger.model import autoModel

"""  

核心训练模块

https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html



# Dump default configuration to have as reference
# 生成配置文件

python  trainer.py --print_config > default_config.yaml

# Create config including only options to modify
nano config.yaml
# Run training using created configuration
#运行训练操作

python trainer.py--config config.yaml



"""

# 更多的配置  https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.utilities.cli.html?highlight=LightningCLI#pytorch_lightning.utilities.cli.LightningCLI
cli = LightningCLI(autoModel,save_config_overwrite=True)