@echo off
setlocal

pushd "%~dp0"
echo prepare installation directory
mkdir "C:\Program Files\Common Files\VST3"
echo remove old file
del /q "C:\Program Files\Common Files\VST3\Wahwth.vst3"
echo copy plugin file
xcopy "Wahwth.vst3" "C:\Program Files\Common Files\VST3" /Y /E /I
popd

echo "インストールが完了しました"
pause
