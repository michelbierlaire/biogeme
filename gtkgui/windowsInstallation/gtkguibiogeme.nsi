!include "EnvVarUpdate.nsh"

Function addToPath
;MessageBox MB_OK $INSTDIR
${EnvVarUpdate} $0 "PATH" "A" "HKLM" $INSTDIR
FunctionEnd

;------------------------------------------------------------
RequestExecutionLevel highest ;Needed to write to shortcut to startup menu

Name "Biogeme Installer"

OutFile "BiogemeInstaller.exe"
DirText "This will install Biogeme on your computer. Choose a directory."

InstallDir $PROGRAMFILES\Biogeme
Icon "smiley.ico"

page directory
page instfiles


;------------------------------------------------------------
Section "Install"

SetOutPath $INSTDIR

File /nonfatal /a /r "tmp\biogeme-inst\*"

call addToPath

File smiley.ico
CreateDirectory "$SMPROGRAMS\biogeme"
CreateShortcut "$SMPROGRAMS\biogeme\biogeme.lnk" "$INSTDIR\gtkguibiogeme_static.exe" "" "$INSTDIR\smiley.ico"
CreateShortcut "$SMPROGRAMS\biogeme\uninstall.lnk" "$INSTDIR\UninstallBiogeme.exe"
WriteUninstaller $INSTDIR\UninstallBiogeme.exe

SectionEnd

;------------------------------------------------------------
Section "Uninstall"

${un.EnvVarUpdate} $0 "PATH" "R" "HKLM" $INSTDIR

Delete "$SMPROGRAMS\biogeme\biogeme.lnk"
Delete "$SMPROGRAMS\biogeme\uninstall.lnk"
rmDir "$SMPROGRAMS\biogeme"

Delete $INSTDIR\UninstallBiogeme.exe
RMDir /r /REBOOTOK $INSTDIR


SectionEnd
