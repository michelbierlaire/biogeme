import os
import shutil
import tomlkit as tk
from faq import faq
from sections import about, install, documentation, archives, resources, special

TARGET_FILE = 'index.html'
DATA_FILE = 'data.toml'
PORTFOLIO_MODAL = 'portfolio_modal.html'
PORTFOLIO_FILE = 'portfolio_grid.html.orig'
PORTFOLIO_GRID_ITEM = 'portfolio_grid_item.html'
HTML_FILE = 'index.html.orig'
FAQ_FILE = 'faq.html'
CARD_FILE = 'card.html'
SPECIAL_FILE = 'special.html'


def replace(orig_text, dictionary):
    for k, v in dictionary.items():
        orig_text = orig_text.replace(k, v)
    return orig_text


def get_section(content):
    all_html = ''
    id = 0
    for card_title, card_paragraphs in content.items():
        with open(CARD_FILE, encoding='utf-8') as f:
            html = f.read()
        text = ''
        for p in card_paragraphs:
            text += '<p class="card-text">'
            text += p
            text += '</p>\n'
        replacements = {
            '__TITLE__': card_title,
            '__CONTENT__': text,
        }
        all_html += replace(html, replacements) + '\n'
    return all_html


def get_faq():
    all_html = ''
    id = 0
    for question, answer in faq.items():
        id += 1
        with open(FAQ_FILE, encoding='utf-8') as f:
            html = f.read()
        replacements = {
            '__ID__': str(id),
            '__QUESTION__': question,
            '__ANSWER__': answer,
        }
        all_html += replace(html, replacements) + '\n'
    return all_html


def get_special(content):
    all_html = ''
    with open(SPECIAL_FILE, encoding='utf-8') as f:
        html = f.read()

    for special_title, special_content in content.items():
        my_html = html
        replacements = {
            '__TITLE__': special_title,
            '__CONTENT__': special_content,
        }
        all_html += replace(my_html, replacements)
    return all_html


def get_portfolio_grid(doc):
    all_html = ''
    for data, values in doc.items():
        with open(PORTFOLIO_GRID_ITEM) as f:
            html = f.read()
        replacements = {
            '__ID__': data,
            '__TITLE__': values['title'],
            '__IMAGE__': values['picture'],
            '__SHORT__': values['short_description'],
            '__LONG__': values['long_description'],
            '__PDF__': values['pdf_file'],
            '__DATA__': values['data_file'],
        }
        all_html += replace(html, replacements) + '\n'
    return all_html


def get_portfolio_modals(doc):
    all_html = ''
    for data, values in doc.items():
        with open(PORTFOLIO_MODAL, encoding='utf-8') as f:
            html = f.read()
        replacements = {
            '__ID__': data,
            '__TITLE__': values['title'],
            '__IMAGE__': values['picture'],
            '__SHORT__': values['short_description'],
            '__LONG__': values['long_description'],
            '__PDF__': values['pdf_file'],
            '__DATA__': values['data_file'],
        }
        all_html += replace(html, replacements) + '\n'
    return all_html


with open(DATA_FILE, encoding='utf-8') as f:
    text = f.read()
doc = tk.parse(text)

with open(HTML_FILE, encoding='utf-8') as f:
    html = f.read()

with open(PORTFOLIO_FILE, encoding='utf-8') as f:
    portfolio_html = f.read()

portfolio_html = replace(
    portfolio_html,
    {
        '__PORTTITLE__': 'Data',
        '__PORTDESC__': (
            'We provide here some choice data sets that can be used '
            'for research and education.'
        ),
        '__CONTENT__': get_portfolio_grid(doc),
    },
)

replacements = {
    '__PORTFOLIO_MODALS__': get_portfolio_modals(doc),
    '__PORTFOLIO__': portfolio_html,
    '__FAQ__': get_faq(),
    '__ABOUT__': get_section(about),
    '__INSTALL_': get_section(install),
    '__DOC__': get_section(documentation),
    '__RES__': get_section(resources),
    '__ARCHIVES__': get_section(archives),
    '__SPECIAL__': get_special(special),
}

html = replace(html, replacements)

# Save the current website just in case
shutil.rmtree('website.old', ignore_errors=True)
if os.path.isdir('website'):
    os.rename('website', 'website.old')
os.mkdir('website')

# Generate the main page
with open(f'website/{TARGET_FILE}', 'w', encoding='utf-8') as f:
    print(html, file=f)

# Copy the Bootstrap js
shutil.copytree('js', 'website/js')
# Copy the Bootstrap css
shutil.copytree('css', 'website/css')
# Copy the Bootstrap assets
shutil.copytree('assets', 'website/assets')


# Copy the PythonBiogeme distribution
shutil.copytree('otherFiles/distrib', 'website/distrib')

# Copy the sphinx documentation
shutil.copytree('../docs/build/html', 'website/sphinx')
