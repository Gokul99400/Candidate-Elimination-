from typing import List

import datasets

import pandas


VERSION = datasets.Version("1.0.0")
_BASE_FEATURE_NAMES = [
    "color",
    "size",
    "act",
    "age",
    "is_inflated"
]


DESCRIPTION = "Balloons dataset from the UCI ML repository."
_HOMEPAGE = "https://archive.ics.uci.edu/ml/datasets/Balloons"
_URLS = ("https://huggingface.co/datasets/mstz/balloons/raw/balloons.csv")
_CITATION = """
@misc{misc_balloons_13,
  title        = {{Balloons}},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: \\url{10.24432/C5BP4D}}
}"""

# Dataset info
urls_per_split = {
    "adult_or_stretch": {"train": "https://huggingface.co/datasets/mstz/balloons/raw/main/adult+stretch.data"},
    "adult_and_stretch": {"train": "https://huggingface.co/datasets/mstz/balloons/raw/main/adult-stretch.data"},
    "yellow_and_small": {"train": "https://huggingface.co/datasets/mstz/balloons/raw/main/yellow-small.data"},
    "yellow_and_small_or_adult_and_stretch": {"train": "https://huggingface.co/datasets/mstz/balloons/raw/main/yellow-small+adult-stretch.data"}
}
features_types_per_config = {
    "adult_or_stretch": {
        "color": datasets.Value("string"),
        "size": datasets.Value("string"),
        "act": datasets.Value("string"),
        "age": datasets.Value("string"),
        "is_inflated": datasets.ClassLabel(num_classes=2)
    },
    "adult_and_stretch": {
        "color": datasets.Value("string"),
        "size": datasets.Value("string"),
        "act": datasets.Value("string"),
        "age": datasets.Value("string"),
        "is_inflated": datasets.ClassLabel(num_classes=2)
    },
    "yellow_and_small": {
        "color": datasets.Value("string"),
        "size": datasets.Value("string"),
        "act": datasets.Value("string"),
        "age": datasets.Value("string"),
        "is_inflated": datasets.ClassLabel(num_classes=2)
    },
    "yellow_and_small_or_adult_and_stretch": {
        "color": datasets.Value("string"),
        "size": datasets.Value("string"),
        "act": datasets.Value("string"),
        "age": datasets.Value("string"),
        "is_inflated": datasets.ClassLabel(num_classes=2)
    }
}
features_per_config = {k: datasets.Features(features_types_per_config[k]) for k in features_types_per_config}


class BalloonsConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(BalloonsConfig, self).__init__(version=VERSION, **kwargs)
        self.features = features_per_config[kwargs["name"]]


class Balloons(datasets.GeneratorBasedBuilder):
    # dataset versions
    DEFAULT_CONFIG = "adult_or_stretch"
    BUILDER_CONFIGS = [
        BalloonsConfig(name="adult_or_stretch",
                    description="Binary classification, balloons are inflated if age == adult or act == stretch."),
        BalloonsConfig(name="adult_and_stretch",
                    description="Binary classification, balloons are inflated if age == adult and act == stretch."),
        BalloonsConfig(name="yellow_and_small",
                    description="Binary classification, balloons are inflated if color == yellow and size == small."),
        BalloonsConfig(name="yellow_and_small_or_adult_and_stretch",
                    description="Binary classification, balloons are inflated if color == yellow and size == small or age == adult and act == stretch.")
    ]


    def _info(self):
        info = datasets.DatasetInfo(description=DESCRIPTION, citation=_CITATION, homepage=_HOMEPAGE,
                                    features=features_per_config[self.config.name])

        return info
    
    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        downloads_per_config = {config: dl_manager.download_and_extract(urls_per_split) for config in urls_per_split}
        print(downloads_per_config)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloads_per_config[self.config.name][self.config.name]["train"]})
        ]
    
    def _generate_examples(self, filepath: str):
        data = pandas.read_csv(filepath, header=None)
        data.columns = _BASE_FEATURE_NAMES
        data.loc[:, "is_inflated"] = data.is_inflated.apply(lambda x: 1 if x == "T" else 0)

        for row_id, row in data.iterrows():
            data_row = dict(row)

            yield row_id, data_row
