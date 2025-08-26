# ~/.bashrc: executed by bash for non-login shells

# If not running interactively, don't do anything
[ -z "$PS1" ] && return

# Set terminal type for color support
export TERM=xterm-256color

# Set colorful prompt
PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '

# Enable color support for common commands
export CLICOLOR=1
export LS_COLORS='di=1;34:ln=35:so=32:pi=33:ex=31:bd=34;46:cd=34;43:su=30;41:sg=30;46:tw=30;42:ow=30;43'
alias ls='ls --color=auto'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'