import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

def execute_notebook(notebook_path):
    # Set environment variables to suppress the debugger warnings
    os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
    os.environ['PYDEVD_WARN_FROZEN'] = 'off'

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600)  # Adjust the timeout as needed (in seconds)

    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
    except Exception as e:
        raise RuntimeError(f"Error executing notebook: {notebook_path}, error: {str(e)}")
    finally:
        # Clean up the environment variables after execution
        os.environ.pop('PYDEVD_DISABLE_FILE_VALIDATION', None)
        os.environ.pop('PYDEVD_WARN_FROZEN', None)

    # Save the executed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    return nb

def export_html(notebook_path, output_dir):
    exporter = HTMLExporter()
    output_notebook, _ = exporter.from_filename(notebook_path)
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(notebook_path))[0] + '.html')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_notebook)

if __name__ == '__main__':
    notebooks_directory = '.'  # Replace with the directory containing your notebooks

    # Optional: Create a directory to store the executed notebook outputs in HTML format
    output_directory = 'output_directory'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List all .ipynb files in the notebooks directory and execute each of them
    for filename in os.listdir(notebooks_directory):
        if filename.endswith('.ipynb'):
            notebook_path = os.path.join(notebooks_directory, filename)
            try:
                execute_notebook(notebook_path)
                print(f"Notebook executed successfully: {notebook_path}")
                export_html(notebook_path, output_directory)  # Export notebook to HTML if needed
            except Exception as e:
                print(f"Error executing notebook: {notebook_path}, error: {str(e)}")
                continue
