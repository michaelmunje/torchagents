sphinx-apidoc -f -o source/ ../torchagents/ --separate
sphinx-build -b html .\source\ .\build\
