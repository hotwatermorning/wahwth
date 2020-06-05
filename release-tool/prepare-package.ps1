# cd into the build directory
cd "$PSScriptRoot"
cd ..\build

# renew a temporary release directory
if (Test-Path tmp_release) {
  Remove-Item -Path tmp_release -Force -Recurse
}
New-Item tmp_release -Force -ItemType Directory

# copy binary files into the temporary release directory
New-Item tmp_release\VST3 -Force -ItemType Directory
Copy-Item -Path .\Wahwth_artefacts\Release\Standalone -Destination tmp_release -Recurse
Copy-Item -Path .\Wahwth_artefacts\Release\VST3\Wahwth.vst3 -Destination tmp_release\VST3 -Recurse

# copy resource files for release
Copy-Item -Path ..\resources\README-ja.pdf -Destination tmp_release
Copy-Item -Path ..\resources\win\Install-Wahwth-VST3.bat -Destination tmp_release\VST3
Copy-Item -Path ..\resources\win\Uninstall-Wahwth-VST3.bat -Destination tmp_release\VST3

# create zip archive
cd tmp_release
Compress-Archive -Path .\* -DestinationPath ..\wahwth-release-win.zip
