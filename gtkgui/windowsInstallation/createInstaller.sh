#!/usr/bin/bash

#cd ../../source

echo "Current directory: $PWD"

tempfiles=gtkgui/windowsInstallation/tmp
mkdir -p $tempfiles

dest=$tempfiles/biogeme-inst
pyt=$dest
gtk=$dest/share
glib=$dest/share/glib-2.0
version=`grep "AC_INIT" configure.ac | cut -d "[" -f3 | cut -d "]" -f1`

echo "Creating standalone python environment."
mkdir -p $dest
mkdir -p $pyt
mkdir -p $pyt/include
mkdir -p $pyt/lib
cp -rf /mingw64/include/python3.5m $pyt/include
cp -rf /mingw64/lib/python3.5 $pyt/lib

echo "Importing python modules for biogeme"
cp -rf pythonbiogeme/python/*.py $dest

echo "Copying icons for GTK"
mkdir -p $gtk
cp -rf /mingw64/share/icons $gtk

echo "Copying schemas for GTK"
mkdir -p $glib
cp -rf /mingw64/share/glib-2.0/schemas $glib

echo "Adding DLLs"
cp /mingw64/bin/libatk-1.0-0.dll $dest
cp /mingw64/bin/libatkmm-1.6-1.dll $dest
cp /mingw64/bin/libbz2-1.dll $dest
cp /mingw64/bin/libcairo-2.dll $dest
cp /mingw64/bin/libcairo-gobject-2.dll $dest
cp /mingw64/bin/libcairomm-1.0-1.dll $dest
cp /mingw64/bin/libepoxy-0.dll $dest
cp /mingw64/bin/libexpat-1.dll $dest
cp /mingw64/bin/libffi-6.dll $dest
cp /mingw64/bin/libfontconfig-1.dll $dest
cp /mingw64/bin/libfreetype-6.dll $dest
cp /mingw64/bin/libgcc_s_seh-1.dll $dest
cp /mingw64/bin/libgdk-3-0.dll $dest
cp /mingw64/bin/libgdk_pixbuf-2.0-0.dll $dest
cp /mingw64/bin/libgdkmm-3.0-1.dll $dest
cp /mingw64/bin/libgio-2.0-0.dll $dest
cp /mingw64/bin/libgiomm-2.4-1.dll $dest
cp /mingw64/bin/libglib-2.0-0.dll $dest
cp /mingw64/bin/libglibmm-2.4-1.dll $dest
cp /mingw64/bin/libgmodule-2.0-0.dll $dest
cp /mingw64/bin/libgobject-2.0-0.dll $dest
cp /mingw64/bin/libgtk-3-0.dll $dest
cp /mingw64/bin/libgtkmm-3.0-1.dll $dest
cp /mingw64/bin/libgraphite2.dll $dest
cp /mingw64/bin/libharfbuzz-0.dll $dest
cp /mingw64/bin/libiconv-2.dll $dest
cp /mingw64/bin/libintl-8.dll $dest
cp /mingw64/bin/libpango-1.0-0.dll $dest
cp /mingw64/bin/libpangocairo-1.0-0.dll $dest
cp /mingw64/bin/libpangoft2-1.0-0.dll $dest
cp /mingw64/bin/libpangomm-1.4-1.dll $dest
cp /mingw64/bin/libpangowin32-1.0-0.dll $dest
cp /mingw64/bin/libpcre-1.dll $dest
cp /mingw64/bin/libpixman-1-0.dll $dest
cp /mingw64/bin/libpng16-16.dll $dest
cp /mingw64/bin/libpython3.5m.dll $dest
cp /mingw64/bin/libsigc-2.0-0.dll $dest
cp /mingw64/bin/libstdc++-6.dll $dest
cp /mingw64/bin/libwinpthread-1.dll $dest
cp /mingw64/bin/zlib1.dll $dest


echo "Copying static executable and batch file"
cp gtkgui/gtkguibiogeme_static.exe $dest
cp gtkgui/gtkbiogeme.ui $dest
#cp ../release_scripts/windows/ressources/guibiogeme.bat $dest

echo "Calling NSIS to build installer"
makensis gtkgui/windowsInstallation/gtkguibiogeme.nsi > nsis.log

if [ $? = 1 ]
then
  echo "Failed to create installer with NSIS, see nsis.log"
  return 1
else
  echo "Moving installer to windows release folder (PythonBiogeme-${version}-setup.exe)"
  mv gtkgui/windowsInstallation/BiogemeInstaller.exe PythonBiogeme-${version}-installer.exe
fi

echo "Cleaning"
/bin/rm -r $tempfiles
