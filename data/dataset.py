from torchvision import transforms

from .ade_fss import DatasetADE
from .cityscapes_fss import DatasetCITYSCAPES
from .general_fss import DatasetGENERAL
from .pascalvoc_fss import DatasetVOC


class PromptingDataset:
    @classmethod
    def initialize(cls, img_size, datapath):
        cls.datasets = {
            "cityscapes": DatasetCITYSCAPES,
            "pascal": DatasetVOC,
            "ade20k": DatasetADE,
            "lovedarural": DatasetGENERAL,
            "lovedaurban": DatasetGENERAL,
            "mhpv1": DatasetGENERAL,
            "pidray": DatasetGENERAL,
            "houseparts": DatasetGENERAL,
            "pizza": DatasetGENERAL,
            "toolkits": DatasetGENERAL,
            "trash": DatasetGENERAL,
            "uecfood": DatasetGENERAL,
            "zerowaste": DatasetGENERAL,
            "uavid": DatasetGENERAL,
        }

        cls.datapath = datapath
        cls.transform = transforms.Compose(
            [transforms.Resize(size=(img_size, img_size)), transforms.ToTensor()]
        )

    @classmethod
    def build_dataset(cls, benchmark: str, nprompts: int):
        dataset = cls.datasets[benchmark](
            datapath=cls.datapath,
            transform=cls.transform,
            split="val",
            nprompts=nprompts,
            use_original_imgsize=False,
            benchmark=benchmark,
        )
        return dataset

    @classmethod
    def load_support_set(cls, dataset, logger):
        logger.info("Loading support set...")
        support_set = dataset.load_support_set()
        ss_dict = {}
        for idx, support in enumerate(support_set):
            if dataset.ignore_background:
                ss_dict[idx + 1] = support["support_names"]
            else:
                ss_dict[idx] = support["support_names"]
        logger.info(f"Support set: {ss_dict}\n")

        return support_set
