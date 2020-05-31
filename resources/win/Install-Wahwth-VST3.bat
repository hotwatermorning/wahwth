@echo off
setlocal

pushd "%~dp0"
mkdir "C:\Program Files\Common Files\VST3"
del /q "C:\Program Files\Common Files\VST3\Wahwth.vst3"
copy .\Wahwth.vst3 "C:\Program Files\Common Files\VST3"
popd

echo "インストールが完了しました"
pause
