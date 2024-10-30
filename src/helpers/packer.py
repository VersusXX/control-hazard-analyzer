import json
from pathlib import Path
from typing import Dict, Protocol

from src.protocols.collector import DictSI


class IPacker(Protocol):
    def pack(self, out_dir: Path, analyzed_data: Dict[str, DictSI]) -> None: ...


class Packer:
    def pack(self, out_dir: Path, analyzed_data: Dict[str, DictSI], cycle_index: str = None) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for key in analyzed_data:
            if cycle_index:
                with open(out_dir.joinpath("cycle_" + cycle_index + "_" + key + ".data"), "wt") as writter:
                    writter.write(json.dumps(analyzed_data[key]))
            else:
                with open(out_dir.joinpath(key + ".data"), "wt") as writter:
                    writter.write(json.dumps(analyzed_data[key]))
