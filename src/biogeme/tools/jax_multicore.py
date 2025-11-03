#!/usr/bin/env python3
import os
import platform
import warnings


def count_cpu_devices() -> int:
    import jax

    # Count CPU devices
    cpu_devices = [d for d in jax.devices() if d.platform == "cpu"]
    n_cpu_devices = len(cpu_devices)
    return n_cpu_devices


def _platform_quick_command() -> str:
    """Return a platform-specific one-liner to set XLA_FLAGS for the current OS.

    Includes a Jupyter `%env` line for convenience.
    """
    sys = platform.system().lower()
    if sys == "windows":
        # Provide both PowerShell and cmd.exe
        return (
            "Windows PowerShell:\n"
            "  $env:XLA_FLAGS=\"--xla_force_host_platform_device_count=<number_of_cores>\"\n\n"
            "Windows cmd.exe:\n"
            "  set XLA_FLAGS=--xla_force_host_platform_device_count=<number_of_cores>\n\n"
            "Jupyter (new cell, before `import jax`):\n"
            "  %env XLA_FLAGS=\"--xla_force_host_platform_device_count=<number_of_cores>\"\n"
        )
    # Default to POSIX shells for macOS/Linux
    return (
        "macOS / Linux (bash/zsh):\n"
        "  export XLA_FLAGS=\"--xla_force_host_platform_device_count=<number_of_cores>\"\n\n"
        "Jupyter (new cell, before `import jax`):\n"
        "  %env XLA_FLAGS=\"--xla_force_host_platform_device_count=<number_of_cores>\"\n"
    )


def report_jax_cpu_devices() -> str:
    # Count CPU devices
    n_cpu_devices = count_cpu_devices()

    lines = [
        f"Detected CPU devices: {n_cpu_devices} | System logical cores: {os.cpu_count() or 'unknown'}",
        f"Current XLA_FLAGS: {os.environ.get('XLA_FLAGS', '(none set)')}",
        f"Platform: {platform.system()} {platform.release()} | Python: {platform.python_version()}",
        "",
    ]
    return "\n".join(lines)


def warning_cpu_devices() -> None:
    n_cpu_devices = count_cpu_devices()
    if n_cpu_devices <= 1:
        lines = [
            "Note: JAX currently sees 1 CPU device. To parallelize across CPU devices, set XLA_FLAGS as above and restart Python/Jupyter.",
            _platform_quick_command(),
        ]

        warnings.warn("\n".join(lines), stacklevel=2)
    return


if __name__ == "__main__":
    # print(report_jax_cpu_devices())
    warning_cpu_devices()
