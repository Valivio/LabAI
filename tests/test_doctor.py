from __future__ import annotations

import importlib
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock, patch

import labai
from labai.core import config

doctor_module = importlib.import_module("labai.core.doctor")


class FakeDevice:
    def __init__(self, device_type: object, label: str) -> None:
        self.type = device_type
        self.label = label

    def __str__(self) -> str:
        return self.label


class DoctorTests(unittest.TestCase):
    def run_doctor(
        self,
        *,
        configuration: config.LabAIConfig | None = None,
        paths_exist: bool = True,
        python_version_info: tuple[int, int] = (3, 13),
        python_version: str = "3.13.0",
        use_gpu: bool = True,
    ) -> tuple[int, str, Mock, Mock]:
        configuration = configuration or config.LabAIConfig(None, None)
        gpu_type = object()
        device_type = gpu_type if use_gpu else object()
        device = FakeDevice(
            device_type,
            "Device(gpu, 0)" if use_gpu else "Device(cpu, 0)",
        )
        mlx = Mock(gpu=gpu_type)
        mlx.default_device.return_value = device

        output = io.StringIO()
        with (
            patch.object(doctor_module, "mx", mlx),
            patch.object(
                config.os,
                "getenv",
                side_effect=AssertionError("doctor must use load_config"),
            ),
            patch.object(
                doctor_module.config,
                "load_config",
                return_value=configuration,
            ) as load_config,
            patch.object(
                doctor_module.Path,
                "exists",
                return_value=paths_exist,
            ) as exists,
            patch.object(doctor_module.sys, "version_info", python_version_info),
            patch.object(
                doctor_module.platform,
                "python_version",
                return_value=python_version,
            ),
            redirect_stdout(output),
        ):
            exit_code = doctor_module.doctor()

        return exit_code, output.getvalue(), load_config, exists

    def test_reports_success_when_all_checks_pass(self) -> None:
        exit_code, output, load_config, _ = self.run_doctor(
            configuration=config.LabAIConfig(
                data_dir=Path("/synthetic/data"),
                models_dir=Path("/synthetic/models"),
            )
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            output,
            "LabAI doctor\n\n"
            "✓ Python: 3.13.0\n"
            "✓ MLX device: Device(gpu, 0)\n"
            "✓ LABAI_DATA_DIR: OK (/synthetic/data)\n"
            "✓ LABAI_MODELS_DIR: OK (/synthetic/models)\n",
        )
        load_config.assert_called_once_with()

    def test_reports_unconfigured_paths_without_accessing_filesystem(self) -> None:
        exit_code, output, _, exists = self.run_doctor()

        self.assertEqual(exit_code, 1)
        self.assertIn("✗ LABAI_DATA_DIR: not configured\n", output)
        self.assertIn("✗ LABAI_MODELS_DIR: not configured\n", output)
        exists.assert_not_called()

    def test_reports_missing_configured_paths(self) -> None:
        exit_code, output, _, _ = self.run_doctor(
            configuration=config.LabAIConfig(
                data_dir=Path("/synthetic/missing-data"),
                models_dir=Path("/synthetic/missing-models"),
            ),
            paths_exist=False,
        )

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "✗ LABAI_DATA_DIR: missing (/synthetic/missing-data)\n",
            output,
        )
        self.assertIn(
            "✗ LABAI_MODELS_DIR: missing (/synthetic/missing-models)\n",
            output,
        )

    def test_reports_python_and_mlx_failures(self) -> None:
        exit_code, output, _, _ = self.run_doctor(
            configuration=config.LabAIConfig(
                data_dir=Path("/synthetic/data"),
                models_dir=Path("/synthetic/models"),
            ),
            python_version_info=(3, 12),
            python_version="3.12.9",
            use_gpu=False,
        )

        self.assertEqual(exit_code, 1)
        self.assertIn("✗ Python: 3.12.9\n", output)
        self.assertIn("✗ MLX device: Device(cpu, 0)\n", output)


class ConfigTests(unittest.TestCase):
    def test_loads_current_environment_on_every_call(self) -> None:
        with patch.dict(
            os.environ,
            {
                config.DATA_DIR_ENV_VAR: "/synthetic/first-data",
                config.MODELS_DIR_ENV_VAR: "/synthetic/first-models",
            },
            clear=True,
        ):
            first = config.load_config()

            os.environ[config.DATA_DIR_ENV_VAR] = "/synthetic/second-data"
            os.environ[config.MODELS_DIR_ENV_VAR] = "/synthetic/second-models"
            second = config.load_config()

        self.assertEqual(first.data_dir, Path("/synthetic/first-data"))
        self.assertEqual(first.models_dir, Path("/synthetic/first-models"))
        self.assertEqual(second.data_dir, Path("/synthetic/second-data"))
        self.assertEqual(second.models_dir, Path("/synthetic/second-models"))

    def test_expands_home_and_normalizes_relative_paths(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HOME": "/synthetic/home",
                config.DATA_DIR_ENV_VAR: "~/data",
                config.MODELS_DIR_ENV_VAR: "relative-models",
            },
            clear=True,
        ):
            configuration = config.load_config()

        self.assertEqual(configuration.data_dir, Path("/synthetic/home/data"))
        self.assertEqual(
            configuration.models_dir,
            (Path.cwd() / "relative-models").absolute(),
        )
        self.assertTrue(configuration.models_dir.is_absolute())

    def test_missing_environment_variables_produce_none(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            configuration = config.load_config()

        self.assertIsNone(configuration.data_dir)
        self.assertIsNone(configuration.models_dir)

    def test_configuration_is_immutable(self) -> None:
        configuration = config.LabAIConfig(None, None)

        with self.assertRaises(FrozenInstanceError):
            configuration.data_dir = Path("/synthetic/data")  # type: ignore[misc]


class CliBehaviorTests(unittest.TestCase):
    def test_all_current_argument_forms_invoke_doctor(self) -> None:
        argument_forms = (
            ["labai"],
            ["labai", "doctor"],
            ["labai", "--help"],
            ["labai", "unknown", "arguments"],
        )

        for arguments in argument_forms:
            with self.subTest(arguments=arguments):
                with (
                    patch.object(sys, "argv", arguments),
                    patch.object(labai, "doctor", return_value=7) as doctor,
                    self.assertRaises(SystemExit) as raised,
                ):
                    labai.main()

                self.assertEqual(raised.exception.code, 7)
                doctor.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
