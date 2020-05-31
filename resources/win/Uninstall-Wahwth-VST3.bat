@echo off
setlocal

pushd "%~dp0"
del /q "C:\Program Files\Common Files\VST3\Wahwth.vst3"
popd

echo "アンインストールが完了しました"
pause

