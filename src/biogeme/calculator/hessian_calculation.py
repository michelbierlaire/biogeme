import pickle
import subprocess
import sys

import jax

from biogeme.tools import TemporaryFile


def evaluate_hessian_in_subprocess(free_betas_values, data, draws, rv, sum_function):
    """
    Launch a subprocess to compute the Hessian using JAX autodiff.
    """

    import os

    with TemporaryFile() as tmp_file:
        hess_file = tmp_file.fullpath
        print(f"[DEBUG] Created temporary file: {hess_file}")

        args = (free_betas_values, data, draws, rv, sum_function, hess_file)
        payload = pickle.dumps(args)
        cmd = [sys.executable, __file__]
        try:
            result = subprocess.run(
                cmd, input=payload, stderr=subprocess.PIPE, timeout=60
            )
            print(f"[DEBUG] Subprocess return code: {result.returncode}")
            if result.stderr:
                print(f"[DEBUG] Subprocess stderr:\n{result.stderr.decode()}")

            if result.returncode == 0:
                if not os.path.exists(hess_file):
                    print(f"[DEBUG] File does not exist: {hess_file}")
                    return None
                try:
                    with open(hess_file, "rb") as f:
                        print(f"[DEBUG] Reading result from: {hess_file}")
                        return pickle.load(f)
                except (
                    pickle.UnpicklingError,
                    EOFError,
                    AttributeError,
                    FileNotFoundError,
                ) as e:
                    print(f"[DEBUG] Failed to unpickle result: {e}")
                    return None
        finally:
            try:
                os.remove(hess_file)
                print(f"[DEBUG] Deleted temporary file: {hess_file}")
            except FileNotFoundError:
                print(f"[DEBUG] Could not delete missing file: {hess_file}")
        return None


if __name__ == "__main__":
    free_betas_values, data, draws, rv, sum_function, hess_file = pickle.loads(
        sys.stdin.buffer.read()
    )
    print(f"[DEBUG subprocess] Writing result to: {hess_file}")
    fn = lambda p, d, r, rv: sum_function(p, d, r, rv)[0]
    jitted_fn = jax.jit(fn)
    hessian = jax.jacfwd(jax.grad(jitted_fn, argnums=0), argnums=0)(
        free_betas_values, data, draws, rv
    )
    try:
        with open(hess_file, "wb") as f:
            pickle.dump(hessian, f)
            print(f"[DEBUG subprocess] Successfully wrote Hessian to: {hess_file}")
    except Exception as e:
        print(f"[DEBUG subprocess] Failed to write Hessian: {e}")
