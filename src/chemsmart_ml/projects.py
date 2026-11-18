"""Named YAML projects under CHEMSMART's configuration root."""

import logging
import os
from pathlib import Path
from typing import List

import yaml

logger = logging.getLogger(__name__)


def project_directory() -> Path:
    """ML directory under the CHEMSMART configuration root."""
    try:
        from chemsmart.settings.user import CHEMSMARTUserSettings
    except ImportError:
        root = os.environ.get("CHEMSMART_CONFIG_DIR", "~/.chemsmart")
    else:
        root = CHEMSMARTUserSettings.resolve_config_dir()
    return Path(root).expanduser() / "ml"


class ProjectConfig:
    """
    Validated project YAML providing data paths and command defaults.

    Relative paths are resolved against the YAML file's directory.
    Command-line options always take precedence over these defaults.

    Attributes:
        filepath (Path): Resolved YAML path.
        sections (dict): Validated mappings keyed by section name.
    """

    SECTION_KEYS = {
        "data": {"filename", "reference"},
        "split": {"train", "valid", "test"},
        "train": {
            "descriptor",
            "soap_r_cut",
            "soap_n_max",
            "soap_l_max",
            "soap_sigma",
            "soap_aggregation",
            "soap_cache",
            "max_iter",
            "learning_rate",
            "max_leaf_nodes",
            "hidden_size",
            "kernel_alpha",
            "kernel_gamma",
            "kernel_max_samples",
            "kernel_subsample",
            "cv_folds",
            "train_fraction",
            "valid_fraction",
            "test_fraction",
            "target",
            "learning",
            "pairing",
            "features",
            "model",
            "alpha",
            "seed",
            "output",
            "fingerprint",
            "morgan_radius",
            "morgan_size",
            "morgan_chirality",
            "cache",
            "manifest",
            "plot",
        },
        "predict": {"model_path", "output"},
        "validate": {"output", "features", "fingerprint", "pairing"},
        "prepare": {
            "output",
            "morgan_radius",
            "morgan_size",
            "morgan_chirality",
            "cache",
            "manifest",
        },
    }
    PATH_KEYS = (
        "filename",
        "reference",
        "output",
        "model_path",
        "cache",
        "manifest",
        "soap_cache",
    )
    SPLIT_OPTIONS = {
        "train": "train_fraction",
        "valid": "valid_fraction",
        "test": "test_fraction",
    }
    COMMANDS = ("train", "predict", "prepare", "validate")
    SUFFIXES = (".yaml", ".yml")

    def __init__(self, filepath, sections: dict):
        self.filepath = Path(filepath).resolve()
        self.sections = sections

    @classmethod
    def from_file(cls, filepath) -> "ProjectConfig":
        """
        Read and validate a project YAML.

        Raises:
            OSError, yaml.YAMLError: If the file cannot be read.
            ValueError: If sections or keys are invalid.
        """
        filepath = Path(filepath)
        sections = yaml.safe_load(filepath.read_text())
        if not isinstance(sections, dict) or set(sections) - set(
            cls.SECTION_KEYS
        ):
            raise ValueError(
                "Project YAML must contain only "
                f"{'/'.join(cls.SECTION_KEYS)} mappings"
            )
        for section, keys in cls.SECTION_KEYS.items():
            values = sections.setdefault(section, {})
            if not isinstance(values, dict) or set(values) - keys:
                raise ValueError(
                    f"Unknown keys or invalid mapping in {section}"
                )
            cls._resolve_paths(section, values, filepath.parent)
        return cls(filepath, sections)

    @classmethod
    def resolve(cls, value: str) -> Path:
        """Resolve a project name or YAML path to an existing file."""
        path = Path(value).expanduser()
        if path.is_file():
            return path.resolve()
        if path.parent == Path("."):
            root = project_directory()
            if path.suffix in cls.SUFFIXES:
                candidates = [root / path]
            else:
                candidates = [root / (value + s) for s in cls.SUFFIXES]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate.resolve()
        raise ValueError(
            f"Project {value!r} not found; use a YAML path or a name in "
            f"{project_directory()}"
        )

    @classmethod
    def available_names(cls) -> List[str]:
        """Sorted names of projects in the ML project directory."""
        root = project_directory()
        return sorted(
            {
                p.stem
                for suffix in cls.SUFFIXES
                for p in root.glob("*" + suffix)
                if p.is_file()
            }
        )

    @property
    def data(self) -> dict:
        return self.sections["data"]

    @property
    def command_defaults(self) -> dict:
        """Default option values for each subcommand."""
        defaults = {name: dict(self.sections[name]) for name in self.COMMANDS}
        train = defaults["train"]
        for section, option in self.SPLIT_OPTIONS.items():
            if section in self.sections["split"]:
                if option in train:
                    raise ValueError(
                        f"Specify split.{section} or train.{option}, "
                        "not both"
                    )
                train[option] = self.sections["split"][section]
        return defaults

    @classmethod
    def _resolve_paths(cls, section: str, values: dict, root: Path):
        for key in cls.PATH_KEYS:
            if key in values:
                if not isinstance(values[key], str):
                    raise ValueError(f"{section}.{key} must be a path string")
                path = Path(values[key]).expanduser()
                values[key] = str((root / path).resolve())
