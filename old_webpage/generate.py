"""Script generating the webpages for Biogeme

:author: Michel Bierlaire
:date: Sat Aug 20 17:20:35 2022
"""
import os
import shutil
import fileinput
import time
import ast
import pathlib


URLBINDER = (
    'https://mybinder.org/v2/gh/michelbierlaire/biogeme/master?filepath='
)

URLNOTO = (
    'https://noto.epfl.ch/hub/user-redirect/git-pull?'
    'repo=https://github.com/michelbierlaire/biogeme'
    '&urlpath=tree/biogeme/'
)

URLGITHUB = 'https://github.com/michelbierlaire/biogeme/blob/master/'


def replace_in_file(file_name, keyword, content):
    """Replace a string by another one in a file

    :param file_name: name of the file
    :type file_name: str

    :param keyword: keyword to search in the file.
    :type keyword: str

    :param content: text replacing the keyword.
    :type content: str
    """
    for line in fileinput.input(file_name, inplace=True):
        print(line.replace(keyword, content), end='')


def copy_file(name, adir='sources'):
    """Copy a file to the website directory

    :param name: name of the file
    :type name: str

    :param adir: directory where the file is located.
    :type adir: str
    """
    orig = f'{adir}/{name}'
    dest = f'website/{name}'
    shutil.copy(orig, dest)


def copy_example(name, adir, hfiles):
    """Copy an example into the website directory

    :param name: name of the .py file containing the example.
    :type name: str

    :param adir: name of the directory where the example file is.
    :type adir: str

    :param hfiles: list of HTML files that are associated with the example.
    :type hfiles: list(str)
    """
    notebook_fname = f'{name.split(".")[0]}.ipynb'
    destdir = f'website/examples/{adir}'
    orig = f'../examples/{adir}/{name}'
    dest = f'{destdir}/{name}'
    orig_notebook = f'../examples/{adir}/{notebook_fname}'
    dest_notebook = f'{destdir}/{notebook_fname}'
    if not os.path.exists(destdir):
        pathlib.Path(destdir).mkdir(parents=True, exist_ok=True)
    shutil.copy(orig, dest)
    shutil.copy(orig_notebook, dest_notebook)
    for h_file in hfiles:
        orig_html = f'../examples/{adir}/{h_file}'
        dest_html = f'{destdir}/{h_file}'
        shutil.copy(orig_html, dest_html)
    print(f'Copied to {dest}')


def generate_menu(items):
    """Generate the menu for the webage.

    :param items: a dictionary where the keys are the name of the HTML
        file (without extension), and the values are the labels
        appearibng on the menu.
    :type items: dict(str: str)
    """
    file = 'sources/menu.html.orig'
    with open(file, 'w') as menu_f:
        print('<ul class="biomenu">', file=menu_f)
        for html, text in items.items():
            print(
                f'<li class=\'biomenu\'>'
                f'<a class=\'{html}ACTIVE\' href=\'{html}.html\'>'
                f'{text}</a></li>',
                file=menu_f,
            )
        print('</ul>', file=menu_f)


def generate_page(name):
    """Generate a HTML page

    :param name: name of the HTML file, without extension.
    :type name: str
    """
    orig = f'sources/{name}.html.orig'
    dest = f'website/{name}.html'
    shutil.copy(orig, dest)
    menu = 'sources/menu.html.orig'
    menudata = open(menu, 'r').read()
    replace_in_file(dest, 'INCLUDEMENU', menudata)
    banner = 'sources/banner.html.orig'
    bannerdata = open(banner, 'r').read()
    replace_in_file(dest, 'BANNER', bannerdata)
    header = 'sources/header.html.orig'
    headerdata = open(header, 'r').read()
    replace_in_file(dest, 'HEADER', headerdata)
    footer = 'sources/footer.html.orig'
    footerdata = open(footer, 'r').read()
    replace_in_file(dest, 'FOOTER', footerdata)
    replace_in_file(dest, '__DATE', time.strftime('%c'))
    replace_in_file(dest, f'{name}ACTIVE', 'active')


def generate_examples():
    """Generates the HTML page for the examples"""
    dest = 'website/examples.html'
    i = 0
    all_files = {}
    html = {}
    ignore_directory = ['workingNotToDistribute']
    ignore = ['.DS_Store']
    with os.scandir('../examples') as root_dir:
        for path in root_dir:
            if path.is_dir(follow_symlinks=False):
                i += 1
                with os.scandir(path.path) as local:
                    if not path.path in ignore_directory:
                        print(f'----- {path.path} -----')
                        the_f = []
                        the_h = []
                        for file in local:
                            if file.name.endswith('html'):
                                the_h += [file.name]
                            if file.is_file() and file.name.endswith('py'):
                                if not file.name in ignore:
                                    print(f'Parse {file.name}')
                                    # Parse the docstrings
                                    with open(file.path) as the_file:
                                        source_code = the_file.read()
                                    parsed_code = ast.parse(source_code)
                                    the_f += [
                                        (
                                            file.name,
                                            ast.get_docstring(parsed_code),
                                        )
                                    ]

                        all_files[path.name] = the_f
                        html[path.name] = the_h

    all_examples = [
        ('Swissmetro', 'swissmetro'),
        ('Calculating indicators', 'indicators'),
        ('Monte-Carlo integration', 'montecarlo'),
        ('Choice models with latent variables', 'latent'),
        ('Choice models with latent variables: another example', 'latentbis'),
        ('Assisted specification', 'assisted'),
    ]

    result = table_of_contents(all_examples)

    for i in all_examples:
        result += one_example(i[0], i[1], all_files, html)

    result += jupyter_examples()

    replace_in_file(dest, 'THEEXAMPLES', result)


def generate_all_pages(the_pages):
    """Generate all pages of the website

    :param the_pages: a dictionary where the keys are the name of the HTML
        file (without extension), and the values are the labels
        appearibng on the menu.
    :type the_pages: dict(str: str)
    """
    for page in the_pages:
        generate_page(page)
    generate_examples()


# Save the current website just in case
shutil.rmtree('website.old', ignore_errors=True)
if os.path.isdir('website'):
    os.rename('website', 'website.old')
os.mkdir('website')

# Copy the PythonBiogeme distribution
shutil.copytree('otherFiles/distrib', 'website/distrib')

# Copy the sphinx documentation
shutil.copytree('../sphinx/_build/html', 'website/sphinx')

imageFiles = [
    'pandasBiogemeLogo.png',
    'up-arrow.png',
    'github.png',
    'jupyter.png',
    'binder.png',
    'background.jpg',
    'youtube.png',
    'getacrobat.png',
    'pdf-icon.png',
    'dataIcon.png',
]

for img_f in imageFiles:
    copy_file(
        img_f, '/Users/michelbierlaire/ToBackupOnGoogleDrive/webpages/images'
    )

cssFiles = ['biomenu.css', 'biopanel.css', 'biobacktotop.css']
for css_f in cssFiles:
    copy_file(
        css_f, '/Users/michelbierlaire/ToBackupOnGoogleDrive/webpages/css'
    )

jsFiles = ['backtotop.js', 'os.js', 'menu.js']
for js_f in jsFiles:
    copy_file(js_f, '/Users/michelbierlaire/ToBackupOnGoogleDrive/webpages/js')


def clean_doc(doc):
    """Enhance the description of the documentation of an example file.

    :param doc: original documentation.
    :type doc: str

    :return: enhanced documentation.
    :rtype: str
    """
    if doc is None:
        return ''
    doc = doc.replace(':author:', 'Author:')
    doc = doc.replace(':date:', 'Date:')
    doc = doc.replace('\n', '<br>')
    return doc


def table_of_contents(all_examples):
    """Generate the table of contents for the examples

    :param all_examples: list of examples. Each item in the list is a
        tuple with two elements: the name of the example category, and
        the name of the directory where the files are located.
    :type all_examples: list(tuple(str, str))

    :return: table of content
    :rtype: str
    """
    result = ''
    result += '<div class="panel panel-default">'
    result += '<div class="panel-heading">Categories of examples</div>'
    result += '<div class="panel-body">'
    result += '<p>The examples are grouped into the following categories:</p>'
    result += '<ul>'

    for i in all_examples:
        result += f'<li><a href="#{i[1]}">{i[0]}</a></li>'
    result += '<li><a href="#jupyter">Modules illustrations</a></li>'
    result += '</ul>'
    result += (
        '<p>For each example, you have access to the following resources:</p>'
    )
    result += '<ul>'
    result += '<li>A short description extracted from the file comments.</li>'
    result += (
        '<li>Click on the name of the <code>.py</code> file to access the '
        'source code.</li>'
    )
    result += (
        '<li>Click on <img src="github.png" height="30"> to see the notebook '
        'on Github.</li>'
    )
    result += (
        '<li>Click on <img src="jupyter.png" height="30"> to run the notebook '
        'on <a href="http://noto.epfl.ch">noto.epfl.ch</a> '
        '(registration required).</li>'
    )
    result += (
        '<li>Click on <img src="binder.png" height="30"> to run the notebook '
        'on <a href="http://mybinder.org">binder</a> (no registration '
        'required).</li>'
    )
    result += (
        '<li>When available, estimation results are available in one or '
        'several html files.</li>'
    )
    result += '</ul>'
    result += '</div>'
    result += '<div class="panel-footer">'
    result += 'Download the data files from <a href="data.html">here</a>'
    result += '</div>'
    result += '</div>'

    return result


def one_example(name, the_dir, all_files, html):
    """Generate the HTML code for one example

    :param name: name of the example category.
    :type name: str

    :param the_dir: name of the directory where the file is
    :type the_dir: str

    :param all_files: dictionary mapping a directory with the name of the
        relevant .py files.
    :type all_files: dict(str:str)

    :param html: dictionary mapping a directory with the name of the
        relevant .html files.
    :type html: dict(str:str)
    """
    result = f'<a id="{the_dir}"></a>'
    result += '<div class="panel panel-default">'
    result += f'<div class="panel-heading">{name}</div>'
    result += '<div class="panel-body">'
    files = sorted(all_files[the_dir])
    html_files = html[the_dir]
    result += '<table border="1">'
    for i, doc in files:
        main = i.split(".")[0]
        print(f'Example {i}')
        if main.endswith('allAlgos'):
            hfiles = [h for h in html_files if h.startswith(main)]
        else:
            hfiles = [h for h in html_files if h.split(".")[0] == main]
        print(f'HTML for {main}: {hfiles}')
        copy_example(i, the_dir, hfiles)
        fname = f'examples/{the_dir}/{i}'
        result += '<tr>'
        result += (
            f'<td style="text-align:left; background-color:lightgrey">'
            f'<a href="{fname}" target="_blank"><strong>{i}</strong></a></td>'
        )
        result += '<td>'
        result += (
            f'<a href="{URLGITHUB}examples/{the_dir}/{main}.ipynb" '
            f'target="_blank">'
            f'<img src="github.png" height="30"></a>&nbsp;'
        )
        result += (
            f'<a href="{URLNOTO}examples/{the_dir}/{main}.ipynb" target="_blank">'
            f'<img src="jupyter.png" height="30"></a>&nbsp;'
        )
        result += (
            f'<a href="{URLBINDER}examples/{the_dir}/{main}.ipynb" '
            f'target="_blank">'
            f'<img src="binder.png" height="30"></a>'
        )
        result += '</td>'
        result += '</tr>'
        if not hfiles:
            result += '<tr>'
            result += '<td></td>'
            result += f'<td>{clean_doc(doc)}</td>'
            result += '</tr>'
        for k, h_file in enumerate(hfiles):
            h_fname = f'examples/{the_dir}/{h_file}'
            result += '<tr>'
            result += (
                f'<td><a href="{h_fname}" target="_blank">{h_file}</a></td>'
            )
            if k == 0:
                result += f'<td rowspan="{len(hfiles)}">{clean_doc(doc)}</td>'
            result += '</tr>'
    result += '</table>'
    result += '</div>'
    result += '</div>'
    return result


def jupyter_examples():
    """Generates the HTML code for the Jupyter notebooks

    :return: HTML code
    :rtype: str
    """
    exclude = ['.DS_Store', 'notebooks.zip']
    result = '<a id="jupyter"></a>'
    result += '<div class="panel panel-default">'
    result += '<div class="panel-heading">Modules illustrations</div>'
    result += '<div class="panel-body">'
    result += (
        '<p>The following Jupyter notebooks contain illustrations of the '
        'use of the different modules available in the Biogeme package. They '
        'are designed for programmers who are interested to exploit the '
        'functionalities of Biogeme.</p>'
    )
    result += (
        '<p>Consult also the <a href="sphinx/index.html" target="_blank">'
        'documentation of the code</a>.</p>'
    )
    result += '<table>'
    with os.scandir('../examples/notebooks') as root_dir:
        for file in root_dir:
            if (
                file.is_file()
                and file.name not in exclude
                and file.name.endswith('ipynb')
            ):
                print(f'Notebook: {file.name}')
                result += '<tr>'
                result += f'<td>{file.name}</td>'
                result += (
                    f'<td style="text-align:center"><a href="{URLGITHUB}'
                    f'examples/notebooks/{file.name}" target="_blank">'
                    f'<img src="github.png" height="30"></a></td>'
                )
                result += (
                    f'<td style="text-align:center"><a href="{URLNOTO}'
                    f'examples/notebooks/{file.name}" target="_blank">'
                    f'<img src="jupyter.png" height="30"></a></td>'
                )
                result += (
                    f'<td style="text-align:center"><a href="{URLBINDER}'
                    f'examples/notebooks/{file.name}" target="_blank">'
                    f'<img src="binder.png" height="30"></a></td>'
                )
    result += '</table>'
    result += '<div class="panel-footer">'
    result += (
        '<p>Register on noto.epfl.ch '
        '<a href="http://noto.epfl.ch">here</a></p>'
    )
    result += '</div>'
    result += '</div>'
    return result


pages = {
    'index': 'Home',
    'start': 'Start',
    'help': 'Help',
    'install': 'Install',
    'data': 'Data',
    'videos': 'Videos',
    'documents': 'Documentation',
    'examples': 'Examples',
    'courses': 'Courses',
    'software': 'Other softwares',
    'archives': 'Archives',
}

generate_menu(pages)
generate_all_pages(pages)
