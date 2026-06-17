# Sphinx tutorial

## 1. Install

```bash
pip install sphinx myst-parser sphinx-rtd-theme sphinx-design
```

## 2. Build the HTML

From the project root:

```bash
cd /home/paul/Documents/sphinx
make html
```

The result is written to `build/html/`.

## 3. Open in the browser

```bash
xdg-open build/html/index.html   # Linux
open build/html/index.html        # macOS
start build\html\index.html       # Windows
```

or just open the document in Firefox, Chrome etc.. 

## 4. Edit → rebuild → refresh

1. Edit a file in `source/`.
2. Run `make html` again.
3. Refresh the browser tab.

---

## Writing content

You can look at the existing chapters in [source/chapters/](source/chapters/)
as examples of how to structure your own pages (headings, code blocks,
admonitions, tables, etc.).

Add your new `.md` file to the `toctree` in
[source/index.rst](source/index.rst), so that Sphinx picks it up.
