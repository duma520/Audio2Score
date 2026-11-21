cls
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist dist rmdir /s /q dist_pyside6

nuitka --standalone ^
    --enable-plugin=pyside6 ^
    --windows-console-mode=disable ^
    --windows-icon-from-ico=icon.ico ^
    --include-package-data=pypinyin ^
    --include-package=PySide6 ^
    --include-qt-plugins=all ^
    --follow-imports ^
    --jobs=4 ^
    --output-dir=dist_pyside6 ^
    your_script.py

