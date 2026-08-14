@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
if "%PYTHON%" == "" (
	set PYTHON=python
)
set SOURCEDIR=source
set BUILDDIR=build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

if "%1" == "clean" goto clean
if "%1" == "check-code" (
	%PYTHON% create_code_rst.py --check
	goto end
)

%PYTHON% create_code_rst.py --force

if "%1" == "html-fast" (
	set BIOGEME_DOCS_GALLERY_PROFILE=none
	set TARGET=html
) else (
	set BIOGEME_DOCS_GALLERY_PROFILE=full
	set TARGET=%1
)

%SPHINXBUILD% -M %TARGET% %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
set BUILD_STATUS=%ERRORLEVEL%
%PYTHON% ..\tools\clean_example_artifacts.py --apply
set CLEANUP_STATUS=%ERRORLEVEL%
if exist "%SOURCEDIR%\auto_examples" rmdir /s /q "%SOURCEDIR%\auto_examples"
if not "%BUILD_STATUS%" == "0" exit /b %BUILD_STATUS%
if not "%CLEANUP_STATUS%" == "0" exit /b %CLEANUP_STATUS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:clean
%SPHINXBUILD% -M clean %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
%PYTHON% ..\tools\docs_examples.py clean --apply
%PYTHON% ..\tools\clean_example_artifacts.py --apply

:end
popd
