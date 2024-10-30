import logging
import shlex
import shutil
from pathlib import Path
from pprint import pformat
from typing import Dict, Any
import numpy as np

from src.analyzers.gemAnalyzer import GemAnalyzer
from src.analyzers.perfAnalyzer import PerfAnalyzer
from src.analyzers.sshAnalyzer import SshAnalyzer
from src.helpers.backGroundBuilder import BGBuilder
from src.helpers.builder import Builder
from src.helpers.configurator import TestsToMutate
from src.helpers.packer import Packer
from src.protocols.analyzer import Analyzer
from src.protocols.collector import DictSI
from src.protocols.utility import Utility


class Analyze(Utility):
    """Class for configuring and executing test analysis based on the parameters passed to the 'analyze' command,
    creating directories for analysis, performing the analysis, then writing the results
    """

    def __init__(self) -> None:
        self.analyzer: Analyzer | None = None
        self.packer: Packer | None = None
        self.builder: Builder | None = None
        self.test_dir: Path | None = None
        self.analyze_dir: Path | None = None
        self.settings: Dict[str, Any] | None = None
        self.mutation_cycles: int | None = None
        self.tests_to_mutate: TestsToMutate | None = None
        self.prev_cycle_results: Dict[str, float] | None = None
        self.logger = logging.getLogger(__name__)

    def configurate(self, settings: Dict[str, Any]) -> None:
        """Initialize local variables with passed parameters

        :param settings: Passed parameters to the command
        """
        # packer is not passed with command line arguments because there is currently only one implementation of packer
        self.packer = Packer()
        self.settings = settings
        self.logger.setLevel(self.settings["log_level"])
        self.test_dir = Path(settings["test_dir"])
        self.analyze_dir = Path(settings["out_dir"])
        self.settings["compiler_args"] = shlex.split(settings["compiler_args"])
        self.builder = Builder(self.settings)
        self.mutation_cycles = settings["mutation_cycles"]
        self.tests_to_mutate = settings["tests_to_mutate"]
        if settings.get("prev_cycle_results"):
            self.prev_cycle_results = settings["prev_cycle_results"]
        elif self.mutation_cycles > 0:
            self.prev_cycle_results = {}
        match settings["profiler"]:
            case "perf":
                self.analyzer = PerfAnalyzer(self.builder, settings)
            case "gem5":
                self.analyzer = GemAnalyzer(self.builder, settings)
            case "ssh":
                self.analyzer = SshAnalyzer(BGBuilder(settings, self.builder), settings)
            case _:
                raise Exception(f'"{settings["profiler"]}" is unknown profiler')

    def run(self) -> None:
        """Print log info, then check that input tests directory's path is passed,
        and output directory's path also passed, create directories for analysis, start the analysis,
        and pack the results
        If mutation_cycles > 0, we need to further mutate the tests, and reanalyze them. Firstly we detect cycle index.
        To decide which tests to mutate, we check tests_to_mutate parameter. If it is set to "WORST_BP_RESULT", we will
        choose tests with the worst BP result (highest BP incorrect percentage). Otherwise, we will choose all tests.
        And chosen tests will be mutated and reanalyzed.
        """
        self.logger.info("Analyze running. Settings:")
        self.logger.info(pformat(self.settings))
        if self.analyze_dir is None or self.test_dir is None:
            self.logger.warn("analyze_dir or test_dir are partially unknown. Exiting...")
            return
        self.create_empty_dir(self.analyze_dir)
        data = self.analyze(self.test_dir)
        self.fin_analyzer()
        if self.mutation_cycles > 0:
            if len(self.prev_cycle_results.items()) == 0:
                cycle_index = "0"
            else:
                cycle_index = str(max(int(key.split("_")[1]) for key in self.prev_cycle_results.keys()) + 1)
            self.mutation_cycles -= 1
            self.pack(self.analyze_dir, data, cycle_index)

            if self.tests_to_mutate == TestsToMutate.WORST_BP_RESULT:
                self.get_test_with_most_bp_incorrect(data, cycle_index)
            else:
                self.get_all_tests(data, cycle_index)

            self.logger.debug(f"Cycle {cycle_index} results:")
            for key, value in self.prev_cycle_results.items():
                self.logger.debug(f"{key}: {value}")

            # TODO: Implement proper mutation of tests
            """while mutation is not implemented, we will "imagine" that tests are mutated, and analyze them again in
            new cycle, and collect the results """
            if self.mutation_cycles > 0:
                self.run()
        else:
            self.pack(self.analyze_dir, data)

    def get_test_with_most_bp_incorrect(self, data: Dict[str, Dict[str, int]], cycle_index: str):
        """Get the test with the highest BP incorrect percentage and add it to the results by cycle index"""

        def calculate_bp_incorrect_percentage(src_data: Dict[str, int]) -> float:
            bp_lookups = src_data.get(
                "branchPred.lookups",
                src_data.get("branchPred.btb.lookups::total", np.nan),
            )
            bp_incorrect = src_data.get("branchPred.condIncorrect", np.nan)
            return round((bp_incorrect / float(bp_lookups) * 100 if bp_lookups != 0 else 0), 2)

        best_test_name, best_test_bp_incorrect_percentage = max(
            ((src_file, calculate_bp_incorrect_percentage(src_data)) for src_file, src_data in data.items()),
            key=lambda x: x[1],
            default=("", 0),
        )
        self.prev_cycle_results["cycle_" + cycle_index + "_" + best_test_name] = best_test_bp_incorrect_percentage

    def get_all_tests(self, data: Dict[str, Dict[str, int]], cycle_index: str):
        """Get all tests' BP incorrect percentage and add them to the results by cycle index"""

        for src_file, src_data in data.items():
            bp_lookups = src_data.get(
                "branchPred.lookups",
                src_data.get("branchPred.btb.lookups::total", np.nan),
            )
            bp_incorrect = src_data.get("branchPred.condIncorrect", np.nan)
            bp_incorrect_percentage = round((bp_incorrect / float(bp_lookups) * 100 if bp_lookups != 0 else 0), 2)
            self.prev_cycle_results["cycle_" + cycle_index + "_" + src_file] = bp_incorrect_percentage

    def create_empty_dir(self, dir_path: Path) -> None:
        """Ensure the specified directory is empty by removing it if it exists and then creating a new empty directory

        :param dir_path: The path to the directory to be created
        """
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True)

    def analyze(self, test_dir: Path) -> Dict[str, DictSI]:
        """Execute and analyze tests from the specified test directory

        :param test_dir: The directory containing the test files to be analyzed
        :return: A dictionary containing the results of the analysis
        """
        print(f"[+]: Execute and analyze tests from {test_dir.absolute().as_posix()}")
        if self.analyzer is None:
            self.logger.warn("Analyzer is not provided.")
            return {}
        return self.analyzer.analyze(test_dir)

    def fin_analyzer(self) -> None:
        """Finalize the analyzer, performing any necessary cleanup actions"""
        if self.analyzer is not None:
            self.analyzer.fin()

    def pack(self, analyze_dir: Path, analyzed_data: Dict[str, DictSI], cycle_index: str = None) -> None:
        """Save the results of the analysis to the specified directory

        :param analyze_dir: The directory where the analysis results will be saved
        :param analyzed_data: The data resulting from the analysis
        :param cycle_index: The index of the current cycle of analysis
        """
        print(f"[+]: Save analysis' results to {analyze_dir.absolute().as_posix()}")
        self.packer.pack(analyze_dir, analyzed_data, cycle_index)
