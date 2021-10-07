import logging
import pathlib
import sys
from typing import Dict, Tuple

from mmdet.utils import collect_env
from omegaconf.omegaconf import DictConfig, OmegaConf

from mmcv import Config

# If you use try instead of if, it will raise mypy error,
# For detail please check following;
# https://www.gitmemory.com/issue/python/mypy/9856/752267294
if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final

logger = logging.getLogger(__name__)


class ConfigConverter:
    @classmethod
    def from_hydra(
        cls, hydra_config: DictConfig, cwd: pathlib.Path
    ) -> Tuple[Config, Dict]:
        """Convert hydra config to mmcv config.

        Args:
            hydra_config (DictConfig): A hydra config.
            cwd (pathlib.Path): A path of current working directory
                aquired by running `hydra.utils.get_original_cwd()`.

        Returns:
            Config: A mmcv config which is conveted from hydra config.
            Dict: A meta data used in mmcv.

        Raises:
            ValueError: If `hydra_config` does not have the attribute
                `mmcv_config_path`.

        """
        OmegaConf.set_readonly(hydra_config, True)

        try:
            mmcv_config_path: Final = cwd / hydra_config.mmcv_config_path
        except Exception:
            logging.error("config should have attribute `mmcv_config_path`.")
            raise ValueError()

        mmcv_config = Config.fromfile(str(mmcv_config_path))

        # Because hydra automatically change the working directory,
        # orginal cwd should be added to the path related configs.
        mmcv_config.data.train.ann_file = str(cwd / mmcv_config.data.train.ann_file)
        mmcv_config.data.val.ann_file = str(cwd / mmcv_config.data.val.ann_file)
        mmcv_config.data.test.ann_file = str(cwd / mmcv_config.data.test.ann_file)

        mmcv_config.data.train.img_prefix = str(cwd / mmcv_config.data.train.img_prefix)
        mmcv_config.data.val.img_prefix = str(cwd / mmcv_config.data.val.img_prefix)
        mmcv_config.data.test.img_prefix = str(cwd / mmcv_config.data.test.img_prefix)

        # Convets from hydra to mmcv
        mmcv_config.exp_name = mmcv_config_path.name
        mmcv_config.seed = hydra_config.seed
        mmcv_config.work_dir = "."
        mmcv_config.distributed = False
        mmcv_config.gpu_ids = range(1)

        return mmcv_config, cls._get_mmcv_meta(mmcv_config)

    @classmethod
    def _get_mmcv_meta(cls, mmcv_config: Config) -> Dict:
        """Create meta data from mmcv config.

        Args:
            mmcv_config (Config): A mmcv config.

        Returns:
            Dict: A meta data dict.

        """
        meta = dict()

        env_info_dict: Final = collect_env()
        env_info: Final = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
        meta["env_info"] = env_info
        meta["config"] = mmcv_config.pretty_text
        meta["seed"] = mmcv_config.seed
        meta["exp_name"] = mmcv_config.exp_name

        return meta
