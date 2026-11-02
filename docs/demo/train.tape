# Records docs/demo/train.gif from a real run: `vhs docs/demo/train.tape` from the repo root.
Output docs/demo/train.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 420
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 10s
Show
Type `python -m cnn_from_scratch train` Enter
Sleep 5s
