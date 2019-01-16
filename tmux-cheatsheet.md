# Tmux Cheatsheet

## Tmux Sessions

- `tmux ls` - list sessions
- `tmux new -s <name>` - create a new named session
- `tmux a -t <name>` - attach to a named session
- `tmux kill-session -t <name>` - kill a given session
- `bind-key $` - rename a session

## Panes

- `bind-key %` - split current pane vertically
- `bind-key "` - split current pane horizontally
- `bind-key q` - show numeric pane values
- `bind-key o` - cycle through panes
- `bind-key x` - kill current pane (with confirmation)
- `bind-key <arrow keys>` - navigate through panes

## Windows / Tabs

- `bind-key c` - create a new window
- `bind-key w` - list windows/tabs
- `bind-key ,` - rename a window/tab
- `bind-key &` - kill current window (with confirmation)
- `bind-key <number>` - go to numbered window

## Source

Commands are copied from [here](https://www.cheatography.com/thecultofkaos/cheat-sheets/tmux-basics/). I only copied commands that I use frequently.