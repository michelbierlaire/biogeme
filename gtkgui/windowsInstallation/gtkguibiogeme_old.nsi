!include "EnvVarUpdate.nsh"

Function addToPath
;MessageBox MB_OK $INSTDIR
${EnvVarUpdate} $0 "PATH" "A" "HKLM" $INSTDIR
FunctionEnd

;------------------------------------------------------------
RequestExecutionLevel highest ;Needed to write to shortcut to startup menu

Name "Biogeme Installer"

DirText "This will install Biogeme on your computer. Choose a directory."

;------------------------------------------------------------



OutFile "biogemeInstaller.exe"

InstallDir "$PROGRAMFILES\biogeme"
Icon "smiley.ico"

page directory
page instfiles

Section "Install"
SetOutPath $INSTDIR
File libatk-1.0-0.dll
File libatkmm-1.6-1.dll
File libbz2-1.dll
File libcairo-2.dll
File libcairo-gobject-2.dll
File libcairomm-1.0-1.dll
File libepoxy-0.dll
File libexpat-1.dll
File libffi-6.dll
File libfontconfig-1.dll
File libfreetype-6.dll
File libgcc_s_seh-1.dll
File libgdk_pixbuf-2.0-0.dll
File libgdk-3-0.dll
File libgdkmm-3.0-1.dll
File libgio-2.0-0.dll
File libgiomm-2.4-1.dll
File libglib-2.0-0.dll
File libglibmm-2.4-1.dll
File libgmodule-2.0-0.dll
File libgobject-2.0-0.dll
File libgtk-3-0.dll
File libgtkmm-3.0-1.dll
File libharfbuzz-0.dll
File libiconv-2.dll
File libintl-8.dll
File libpango-1.0-0.dll
File libpangocairo-1.0-0.dll
File libpangoft2-1.0-0.dll
File libpangomm-1.4-1.dll
File libpangowin32-1.0-0.dll
File libpcre-1.dll
File libpixman-1-0.dll
File libpng16-16.dll
File libpython3.5m.dll
File libsigc-2.0-0.dll
File libstdc++-6.dll
File libwinpthread-1.dll
File zlib1.dll

File /r share
#File /oname=guibiogeme.exe gtkguibiogeme_static.exe
File /oname=guibiogeme.exe ..\gtkguibiogeme.exe

call addToPath

File smiley.ico
CreateDirectory "$SMPROGRAMS\biogeme"
CreateShortcut "$SMPROGRAMS\biogeme\biogeme.lnk" "$INSTDIR\guibiogeme.exe" "" "$INSTDIR\smiley.ico"
CreateShortcut "$SMPROGRAMS\biogeme\uninstall.lnk" "$INSTDIR\uninstaller.exe"
WriteUninstaller $INSTDIR\uninstaller.exe
SectionEnd

Section "Uninstall"
Delete $INSTDIR\guibiogeme.exe
Delete $INSTDIR\smiley.ico
Delete $INSTDIR\libatk-1.0-0.dll
Delete $INSTDIR\libatkmm-1.6-1.dll
Delete $INSTDIR\libbz2-1.dll
Delete $INSTDIR\libcairo-2.dll
Delete $INSTDIR\libcairo-gobject-2.dll
Delete $INSTDIR\libcairomm-1.0-1.dll
Delete $INSTDIR\libepoxy-0.dll
Delete $INSTDIR\libexpat-1.dll
Delete $INSTDIR\libffi-6.dll
Delete $INSTDIR\libfontconfig-1.dll
Delete $INSTDIR\libfreetype-6.dll
Delete $INSTDIR\libgcc_s_seh-1.dll
Delete $INSTDIR\libgdk_pixbuf-2.0-0.dll
Delete $INSTDIR\libgdk-3-0.dll
Delete $INSTDIR\libgdkmm-3.0-1.dll
Delete $INSTDIR\libgio-2.0-0.dll
Delete $INSTDIR\libgiomm-2.4-1.dll
Delete $INSTDIR\libglib-2.0-0.dll
Delete $INSTDIR\libglibmm-2.4-1.dll
Delete $INSTDIR\libgmodule-2.0-0.dll
Delete $INSTDIR\libgobject-2.0-0.dll
Delete $INSTDIR\libgtk-3-0.dll
Delete $INSTDIR\libgtkmm-3.0-1.dll
Delete $INSTDIR\libharfbuzz-0.dll
Delete $INSTDIR\libiconv-2.dll
Delete $INSTDIR\libintl-8.dll
Delete $INSTDIR\libpango-1.0-0.dll
Delete $INSTDIR\libpangocairo-1.0-0.dll
Delete $INSTDIR\libpangoft2-1.0-0.dll
Delete $INSTDIR\libpangomm-1.4-1.dll
Delete $INSTDIR\libpangowin32-1.0-0.dll
Delete $INSTDIR\libpcre-1.dll
Delete $INSTDIR\libpixman-1-0.dll
Delete $INSTDIR\libpng16-16.dll
Delete $INSTDIR\libpython3.5m.dll
Delete $INSTDIR\libsigc-2.0-0.dll
Delete $INSTDIR\libstdc++-6.dll
Delete $INSTDIR\libwinpthread-1.dll
Delete $INSTDIR\zlib1.dll


RMDir /r $INSTDIR\share

${un.EnvVarUpdate} $0 "PATH" "R" "HKLM" $INSTDIR

Delete "$SMPROGRAMS\biogeme\biogeme.lnk"
Delete "$SMPROGRAMS\biogeme\uninstall.lnk"
rmDir "$SMPROGRAMS\biogeme"

Delete $INSTDIR\uninstaller.exe
RMDir /r /REBOOTOK $INSTDIR


SectionEnd